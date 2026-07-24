"""A reactive grasp-and-lift expert for LeapLiftEnv.

Four phases -- ALIGN, DESCEND, CLOSE, LIFT -- but the transitions between them
are *state* conditions, not timestamps.

That distinction is the whole ballgame for behaviour cloning, and it cost a full
debugging session to learn. The first version of this file ran off a clock:
`if frac < 0.42: descend`. It produced beautiful demonstrations and a policy that
scored 0%. The reason is that a cloned policy has no clock -- it only sees the
state -- so it has to *infer* the phase from proprioception. The instant the real
hand lagged behind its position command, the inferred phase ran slow, the policy
closed its fingers late, and the state machine desynchronised into nonsense.

With state-based transitions the same lag is self-correcting: if the hand is not
yet down, the condition to descend is simply still true, so it keeps descending.
The expert becomes a genuine function of the observation, which is exactly the
function BC is able to represent.
"""

from dataclasses import dataclass

import numpy as np

from wuji_hands.leap_lift import ALL_ACT, EPISODE_STEPS, LeapLiftEnv
from wuji_hands.scene import CUBE_HALF, GRASP_OFFSET_Z, WRIST_MOUNT_Z

FINGERS = ("if", "mf", "rf")

ALIGN, DESCEND, CLOSE, LIFT = 0, 1, 2, 3
PHASE_NAMES = {ALIGN: "align", DESCEND: "descend", CLOSE: "close", LIFT: "lift"}


@dataclass
class GraspParams:
    # open (pre-grasp) hand pose
    o_mcp: float = -0.03282
    o_pip: float = -0.12778
    o_dip: float = 0.0
    o_tcmc: float = 0.50767
    o_taxl: float = 1.01764
    o_tmcp: float = 0.47159
    o_tipl: float = 0.0

    # closed hand pose (found by CEM -- see docs/tuning.md)
    c_mcp: float = 1.15702
    c_pip: float = 1.57680
    c_dip: float = 0.90219
    c_tcmc: float = 0.95197
    c_taxl: float = 1.44125
    c_tmcp: float = 1.42695
    c_tipl: float = 1.32644

    # geometry
    hover_z: float = 0.04803
    grasp_dz: float = 0.03000
    lift_z: float = 0.12

    # state-machine thresholds
    align_tol: float = 0.02921      # xy error that counts as "over the cube"
    height_tol: float = 0.00864     # z error that counts as "at grasp height"
    close_rate: float = 0.16608      # how fast the closing command ramps per step
    grasped_at: float = 0.76147      # closing progress that counts as "grasped"
    # Must sit clear of the closure the hand reads at reset (~0.07), or the
    # expert believes it is already mid-grasp on step 0 and closes on thin air.
    commit_at: float = 0.20         # closure above which we are committed to the grasp
    # Rate limits, applied relative to where the wrist ACTUALLY is. Commanding
    # the goal directly makes the servo slam down and punt the cube across the
    # table; rate-limiting against the measured position keeps the motion smooth
    # while staying a pure function of the observed state.
    descend_rate: float = 0.01519   # metres of wrist_z per control step
    lift_rate: float = 0.010


class ScriptedExpert:
    def __init__(self, env: LeapLiftEnv, params: GraspParams | None = None, noise: float = 0.0,
                 style: bool = False):
        self.env = env
        self.p = params or GraspParams()
        self.noise = noise
        self.style = style
        self.ix = {n: i for i, n in enumerate(ALL_ACT)}
        self._s = None

    def reset(self, rng: np.random.Generator) -> None:
        """Start an episode.

        `style` draws per-episode offsets -- a personal aiming bias, a squeeze
        strength, a preferred grasp height. It defaults to OFF, and that default
        is load-bearing.

        Those offsets are latent variables: they persist for the whole episode
        and nothing in the observation reveals them. So two episodes can present
        the policy with identical observations and demand systematically
        different actions. BC responds by learning the average, and the residual
        is irreducible noise concentrated in exactly the dimensions that decide
        whether a 5.6 cm cube ends up between the fingers. With style noise on,
        closed-loop success swung between 0% and 64% purely on the training seed,
        at identical validation loss. With it off, the expert is a deterministic
        function of the state plus i.i.d. action noise -- so E[a|o] is exactly the
        expert, which is what BC actually converges to.

        Turn it on (`style=True`) if you want to see that failure for yourself.
        It is a good thing to have seen before you meet it in real teleop data,
        where the latent variable is "which operator collected this episode".
        """
        n = self.noise
        st = self.style
        self._s = {
            "xy_bias": rng.normal(0, 0.004 * (1 + 2 * n), size=2) if st else np.zeros(2),
            "yaw_bias": rng.normal(0, 0.05 * (1 + 2 * n)) if st else 0.0,
            "squeeze": 1.0 + rng.normal(0, 0.06 * (1 + n)) if st else 1.0,
            "grasp_dz": self.p.grasp_dz + (rng.normal(0, 0.004 * (1 + n)) if st else 0.0),
            "align_tol": self.p.align_tol * (1 + rng.normal(0, 0.15 * (1 + n)) if st else 1.0),
            "alpha": 0.0,        # closing command, derived from measured closure
            "phase": ALIGN,
            "rng": rng,
        }

    # ------------------------------------------------------------------ phases

    def _cube_yaw(self) -> float:
        q = self.env.data.qpos[3:7]
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _closure_frac(self) -> float:
        """How closed the hand *actually is*, on a 0 (open) to 1 (closed) scale.

        This is read back from the joint positions, so it is something the policy
        can see in its observation too.
        """
        p = self.p
        open_v = (p.o_mcp + p.o_pip) / 2
        closed_v = (p.c_mcp + p.c_pip) / 2
        q = self.env.data.qpos[self.env.hand_qpos]
        now = float(np.mean(q[[0, 2, 4, 6, 8, 10]]))
        return float(np.clip((now - open_v) / max(closed_v - open_v, 1e-6), 0.0, 1.0))

    def _phase(self, s, closed) -> int:
        """Decide the phase from observable state only.

        Every quantity used here -- the cube pose, the grasp centre, how closed
        the fingers are -- is in the observation. That is the point. An earlier
        version tracked the phase with a hidden latch and a hidden ramp counter,
        which meant that around the moment of committing to a grasp the recorded
        actions were bimodal for observationally identical states: sometimes
        "stay open", sometimes "start closing". A cloned policy then either
        chattered between the two or, if you averaged samples, split the
        difference and closed halfway onto nothing.
        """
        env = self.env
        cube, gc = env.cube_pos, env.grasp_centre
        dxy = float(np.linalg.norm(cube[:2] - gc[:2]))
        dz = float(cube[2] + s["grasp_dz"] - gc[2])

        # Hysteresis via a *visible* variable: once the fingers have started to
        # curl, we are committed, and you can see that in the joint angles.
        committed = closed > self.p.commit_at
        if committed and closed >= self.p.grasped_at:
            return LIFT
        if committed or (dxy < s["align_tol"] and abs(dz) < self.p.height_tol):
            return CLOSE
        if dxy < s["align_tol"]:
            return DESCEND
        return ALIGN

    def act(self, step: int | None = None) -> np.ndarray:
        p, s = self.p, self._s
        env = self.env
        closed = self._closure_frac()
        s["phase"] = phase = self._phase(s, closed)

        # The closing command is a rate-limited ramp anchored to the MEASURED
        # closure rather than an internal counter, so it is a function of the
        # observation and it self-corrects if the fingers stall on the cube.
        if phase in (CLOSE, LIFT):
            s["alpha"] = min(1.0, closed + p.close_rate)
        else:
            s["alpha"] = 0.0

        ctrl = np.zeros(len(ALL_ACT))

        # --- hand: open until we are in position, then ramp closed
        a, sq = s["alpha"], s["squeeze"]
        for f in FINGERS:
            ctrl[self.ix[f + "_mcp"]] = _lerp(p.o_mcp, p.c_mcp * sq, a)
            ctrl[self.ix[f + "_pip"]] = _lerp(p.o_pip, p.c_pip * sq, a)
            ctrl[self.ix[f + "_dip"]] = _lerp(p.o_dip, p.c_dip * sq, a)
            ctrl[self.ix[f + "_rot"]] = 0.0
        ctrl[self.ix["th_cmc"]] = _lerp(p.o_tcmc, p.c_tcmc * sq, a)
        ctrl[self.ix["th_axl"]] = _lerp(p.o_taxl, p.c_taxl * sq, a)
        ctrl[self.ix["th_mcp"]] = _lerp(p.o_tmcp, p.c_tmcp * sq, a)
        ctrl[self.ix["th_ipl"]] = _lerp(p.o_tipl, p.c_tipl * sq, a)

        # --- wrist xy/yaw: chase the cube until we commit, then hold station.
        # "Hold station" means command the wrist's own measured pose, not a pose
        # latched at commit time -- again so the command stays a function of what
        # the policy can see.
        if phase in (ALIGN, DESCEND):
            tx, ty = env.cube_pos[:2] + s["xy_bias"]
            yaw = self._cube_yaw() + s["yaw_bias"]
        else:
            tx, ty = env.data.qpos[7], env.data.qpos[8]
            yaw = float(env.data.qpos[10])
        ctrl[self.ix["wrist_x"]] = tx
        ctrl[self.ix["wrist_y"]] = ty
        ctrl[self.ix["wrist_yaw"]] = yaw

        # --- wrist z: hover, descend to the cube, hold, then lift.
        # Every branch is rate-limited around the MEASURED wrist height, so the
        # command is a function of the current state rather than of a schedule.
        z_grasp = CUBE_HALF + s["grasp_dz"] + GRASP_OFFSET_Z - WRIST_MOUNT_Z
        wz_now = float(env.data.qpos[9])
        if phase == ALIGN:
            wz = p.hover_z
        elif phase in (DESCEND, CLOSE):
            # Only the descent is rate-limited -- that is the one phase where
            # arriving too fast punts the cube across the table. Note the limit is
            # relative to the MEASURED height, so if the hand stalls against the
            # cube the command stalls with it instead of winding up.
            wz = _towards(wz_now, z_grasp, p.descend_rate)
        else:
            # The lift is commanded outright. Rate-limiting it against the measured
            # height deadlocks: holding the cube gives the position servo a
            # steady-state error larger than the rate, so the command stops
            # advancing and the hand hangs 2 cm off the table forever.
            wz = p.lift_z
        ctrl[self.ix["wrist_z"]] = wz

        action = env.normalise(ctrl)
        if self.noise > 0:
            rng = s["rng"]
            action = action + rng.normal(0, 0.010 * self.noise, size=action.shape)
            # Wrist noise is larger on purpose: the wrist is closed-loop, so
            # perturbing it produces genuine off-nominal-state -> corrective-action
            # pairs. That is the data a cloned policy needs to stay stable.
            action[-4:] += rng.normal(0, 0.05 * self.noise, size=4)
        return np.clip(action, -1.0, 1.0)


def _lerp(a, b, t):
    return a + (b - a) * t


def _towards(current, goal, rate):
    """Step `current` toward `goal` by at most `rate`."""
    return current + float(np.clip(goal - current, -rate, rate))


def evaluate(params: GraspParams | None = None, n_episodes: int = 20, noise: float = 0.0,
             seed: int = 0, verbose: bool = False, style: bool = False) -> dict:
    """Success rate of the scripted expert -- the ceiling any BC policy is chasing."""
    env = LeapLiftEnv(seed=seed)
    expert = ScriptedExpert(env, params, noise=noise, style=style)
    rng = np.random.default_rng(seed)
    successes, heights = 0, []
    for ep in range(n_episodes):
        env.reset(seed=seed * 1000 + ep)
        expert.reset(rng)
        info = {}
        for t in range(EPISODE_STEPS):
            _, _, done, info = env.step(expert.act(t))
            if done:
                break
        successes += info["success"]
        heights.append(info["cube_height"])
        if verbose:
            print(f"  ep {ep}: {'OK ' if info['success'] else 'FAIL'} "
                  f"h={info['cube_height']:.3f} final_phase={PHASE_NAMES[expert._s['phase']]}")
    env.close()
    return {"success_rate": successes / n_episodes,
            "mean_height": float(np.mean(heights))}


if __name__ == "__main__":
    print(evaluate(n_episodes=25, verbose=True))
