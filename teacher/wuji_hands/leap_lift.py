"""LeapLift-v0 -- pick a cube off the table with a 16-DoF LEAP hand.

Deliberately small and dependency-free: no gym, no wrappers, no registry. The
whole point is that you can read the entire observation/action contract in one
sitting while you are writing a policy against it.

    obs (31,)   [ hand_qpos(16) | wrist_qpos(4) | cube_pos(3) | cube_quat(4)
                  | cube_pos - grasp_centre(3) | finger_closure(1) ]

    act (20,)   normalised joint position targets in [-1, 1]
                [ hand(16) | wrist_x, wrist_y, wrist_z, wrist_yaw ]

Success = cube centre above SUCCESS_HEIGHT at the end of the episode.

There is deliberately no clock in the observation. An earlier version had a
`time_frac` term and it was actively harmful: the expert was a time-indexed
state machine, so the policy learned to read the phase off proprioception, and
the moment the real hand lagged its command the inferred phase ran slow and the
whole sequence desynchronised. The expert is now purely reactive, so everything
the policy needs to choose an action is in the state you can see.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from wuji_hands.scene import (
    CUBE_HALF,
    GRASP_OFFSET_Z,
    WRIST_MOUNT_Z,
    compile_model,
)

CONTROL_DT = 0.05  # 20 Hz policy rate
PHYSICS_DT = 0.002
N_SUBSTEPS = int(round(CONTROL_DT / PHYSICS_DT))
EPISODE_STEPS = 90  # 4.5 s
SUCCESS_HEIGHT = 0.12

OBS_DIM = 31
ACT_DIM = 20

HAND_ACT = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]
WRIST_ACT = ["wrist_x", "wrist_y", "wrist_z", "wrist_yaw"]
ALL_ACT = HAND_ACT + WRIST_ACT


@dataclass
class ResetOptions:
    """Where the cube starts. Widen these to make the task harder."""

    xy_range: float = 0.055
    yaw_range: float = np.pi / 4
    # The hand does NOT always start centred. This matters more than it looks:
    # if the wrist always began at the origin and then converged onto the cube,
    # "command = where the wrist already is" would be a near-perfect predictor of
    # the expert's action, and a cloned policy will happily learn that shortcut
    # instead of learning to look at the cube. It then drifts, because copying
    # your own position is an integrator with nothing pulling it back.
    wrist_xy_range: float = 0.07
    wrist_yaw_range: float = 0.5


class LeapLiftEnv:
    def __init__(self, seed: int = 0, options: ResetOptions | None = None):
        self.model = compile_model()
        self.data = mujoco.MjData(self.model)
        self.options = options or ResetOptions()
        self.rng = np.random.default_rng(seed)

        name2id = mujoco.mj_name2id
        self.act_ids = np.array(
            [name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_act") for n in ALL_ACT]
        )
        assert (self.act_ids >= 0).all(), "actuator name mismatch"
        self.ctrl_low = self.model.actuator_ctrlrange[self.act_ids, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[self.act_ids, 1].copy()

        self.cube_bid = name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.wrist_bid = name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
        # qpos layout: cube free joint (7), then wrist (4), then hand (16)
        self.cube_qpos = slice(0, 7)
        self.wrist_qpos = slice(7, 11)
        self.hand_qpos = slice(11, 27)

        self._step_count = 0
        self._renderer = None

    # ---------------------------------------------------------------- helpers

    def denormalise(self, action: np.ndarray) -> np.ndarray:
        """[-1, 1] -> actuator ctrlrange."""
        a = np.clip(action, -1.0, 1.0)
        return self.ctrl_low + (a + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

    def normalise(self, ctrl: np.ndarray) -> np.ndarray:
        """ctrlrange -> [-1, 1]. Handy for writing scripted policies in joint units."""
        span = np.maximum(self.ctrl_high - self.ctrl_low, 1e-8)
        return np.clip(2.0 * (ctrl - self.ctrl_low) / span - 1.0, -1.0, 1.0)

    @property
    def grasp_centre(self) -> np.ndarray:
        """World position where the fingers and thumb close on each other."""
        p = self.data.xpos[self.wrist_bid].copy()
        p[2] -= GRASP_OFFSET_Z
        return p

    @property
    def finger_closure(self) -> float:
        """0 = fully open, ~1 = closed on the CLOSED_POSE. Observable phase signal."""
        q = self.data.qpos[self.hand_qpos]
        return float(np.clip(np.mean(q[[0, 2, 4, 6, 8, 10]]) / 1.15, -0.5, 2.0))

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_bid].copy()

    def _obs(self) -> np.ndarray:
        d = self.data
        hand_q = d.qpos[self.hand_qpos]
        wrist_q = d.qpos[self.wrist_qpos]
        cube = d.qpos[self.cube_qpos]
        return np.concatenate(
            [hand_q, wrist_q, cube[:3], cube[3:7], self.cube_pos - self.grasp_centre,
             [self.finger_closure]]
        ).astype(np.float32)

    # ------------------------------------------------------------------- api

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        r = self.options
        x, y = self.rng.uniform(-r.xy_range, r.xy_range, size=2)
        yaw = self.rng.uniform(-r.yaw_range, r.yaw_range)
        self.data.qpos[0:3] = [x, y, CUBE_HALF]
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        # Start the hand open and high, from a randomised wrist pose.
        self.data.ctrl[:] = 0.0
        wx, wy = self.rng.uniform(-r.wrist_xy_range, r.wrist_xy_range, size=2)
        wyaw = self.rng.uniform(-r.wrist_yaw_range, r.wrist_yaw_range)
        for name, val in (("wrist_x", wx), ("wrist_y", wy),
                          ("wrist_z", 0.12), ("wrist_yaw", wyaw)):
            i = self.act_ids[ALL_ACT.index(name)]
            self.data.ctrl[i] = val
            self.data.qpos[self.wrist_qpos][ALL_ACT.index(name) - len(HAND_ACT)] = val
        for _ in range(60):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        return self._obs()

    def step(self, action: np.ndarray):
        ctrl = self.denormalise(np.asarray(action, dtype=np.float64))
        self.data.ctrl[self.act_ids] = ctrl
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._obs()

        height = float(self.cube_pos[2])
        lifted = height > SUCCESS_HEIGHT
        # Shaped reward, mostly so you can plot something during debugging. BC
        # ignores it entirely -- it is here for when you graduate to OGPO.
        reach = -np.linalg.norm(self.cube_pos - self.grasp_centre)
        reward = float(0.5 * reach + 4.0 * max(0.0, height - CUBE_HALF) + 10.0 * lifted)

        done = self._step_count >= EPISODE_STEPS
        info = {"success": bool(lifted), "cube_height": height}
        return obs, reward, done, info

    # ---------------------------------------------------------------- render

    def render(self, width: int = 480, height: int = 360, camera: str = "track") -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
