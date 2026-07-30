"""Collect demonstrations with rendered frames, for the pixel-conditioned policy.

Same expert and acceptance rule as wuji_hands.collect, plus one 224x224 frame
per camera per step. Frames go to their own .npy so train.py can memmap them.

    python -m wuji_bc.collect_pixels --episodes 300
"""

import argparse
import time
from pathlib import Path

import numpy as np

from wuji_hands.expert import ScriptedExpert
from wuji_hands.leap_lift import EPISODE_STEPS, LeapLiftEnv

DATA = Path("/home/leo/wuji-hands/data")
DEFAULT_NPZ = DATA / "leap_lift_pixels.npz"
DEFAULT_FRAMES = DATA / "leap_lift_frames.npy"

RENDER_SIZE = 224
CAMERAS = ("policy", "wrist")


def collect(n_episodes: int = 300, noise: float = 0.5, seed: int = 0,
            keep_failures: bool = False, npz_out: Path = DEFAULT_NPZ,
            frames_out: Path = DEFAULT_FRAMES) -> dict:
    env = LeapLiftEnv(seed=seed)
    expert = ScriptedExpert(env, noise=noise)
    rng = np.random.default_rng(seed)

    obs_buf, act_buf, term_buf, ep_buf, frame_buf = [], [], [], [], []
    kept = attempted = 0
    t0 = time.time()

    while kept < n_episodes:
        obs = env.reset(seed=seed * 100_000 + attempted)
        expert.reset(rng)
        attempted += 1

        ep_obs, ep_act, ep_frames = [], [], []
        info = {"success": False}
        for t in range(EPISODE_STEPS):
            action = expert.act(t)
            ep_obs.append(obs)
            ep_act.append(action)
            # Before the step: frame[i] is what the policy sees when choosing action[i].
            ep_frames.append([env.render(RENDER_SIZE, RENDER_SIZE, c) for c in CAMERAS])
            obs, _, done, info = env.step(action)
            if done:
                break

        if not info["success"] and not keep_failures:
            continue

        n = len(ep_obs)
        obs_buf.append(np.asarray(ep_obs, dtype=np.float32))
        act_buf.append(np.asarray(ep_act, dtype=np.float32))
        frame_buf.append(np.asarray(ep_frames, dtype=np.uint8))
        term = np.zeros(n, dtype=bool)
        term[-1] = True
        term_buf.append(term)
        ep_buf.append(np.full(n, kept, dtype=np.int32))
        kept += 1

        if kept % 25 == 0:
            print(f"  {kept}/{n_episodes} episodes  (accept {kept / attempted:.0%}, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    env.close()

    data = {
        "observations": np.concatenate(obs_buf),
        "actions": np.concatenate(act_buf),
        "terminals": np.concatenate(term_buf),
        "episode_ids": np.concatenate(ep_buf),
    }
    frames = np.concatenate(frame_buf)

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_out, **data)
    np.save(frames_out, frames)

    print(f"\nwrote {npz_out}")
    print(f"wrote {frames_out}  {frames.shape} {frames.nbytes / 1e9:.1f} GB  "
          f"cameras {CAMERAS}")
    print(f"  transitions {len(data['observations'])}  episodes {kept}  "
          f"accept rate {kept / attempted:.0%}", flush=True)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--npz-out", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--frames-out", type=Path, default=DEFAULT_FRAMES)
    a = ap.parse_args()
    collect(a.episodes, a.noise, a.seed, a.keep_failures, a.npz_out, a.frames_out)
    import os
    os._exit(0)
