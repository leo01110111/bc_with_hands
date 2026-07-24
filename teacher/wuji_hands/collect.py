"""Roll the scripted expert and save a behaviour-cloning dataset.

Output is a single .npz of flat transitions -- the format the wuji_bc dataset
loader expects:

    observations  (N, 32) float32
    actions       (N, 20) float32
    terminals     (N,)    bool    True on the last step of each episode
    episode_ids   (N,)    int32

Only successful episodes are kept by default. That is a real modelling choice,
not a detail: BC clones whatever you feed it, so a dataset with failures in it
teaches the policy to fail some fraction of the time.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from wuji_hands.expert import ScriptedExpert
from wuji_hands.leap_lift import EPISODE_STEPS, LeapLiftEnv

DEFAULT_OUT = Path("/home/leo/wuji-hands/data/leap_lift_demos.npz")


def collect(n_episodes: int = 500, noise: float = 0.5, seed: int = 0,
            keep_failures: bool = False, out: Path = DEFAULT_OUT, style: bool = False) -> dict:
    env = LeapLiftEnv(seed=seed)
    expert = ScriptedExpert(env, noise=noise, style=style)
    rng = np.random.default_rng(seed)

    obs_buf, act_buf, term_buf, ep_buf = [], [], [], []
    kept = attempted = 0
    t0 = time.time()

    while kept < n_episodes:
        obs = env.reset(seed=seed * 100_000 + attempted)
        expert.reset(rng)
        attempted += 1

        ep_obs, ep_act = [], []
        info = {"success": False}
        for t in range(EPISODE_STEPS):
            action = expert.act(t)
            ep_obs.append(obs)
            ep_act.append(action)
            obs, _, done, info = env.step(action)
            if done:
                break

        if not info["success"] and not keep_failures:
            continue

        n = len(ep_obs)
        obs_buf.append(np.asarray(ep_obs, dtype=np.float32))
        act_buf.append(np.asarray(ep_act, dtype=np.float32))
        term = np.zeros(n, dtype=bool)
        term[-1] = True
        term_buf.append(term)
        ep_buf.append(np.full(n, kept, dtype=np.int32))
        kept += 1

        if kept % 50 == 0:
            rate = kept / max(attempted, 1)
            print(f"  {kept}/{n_episodes} episodes  (accept {rate:.0%}, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    env.close()
    data = {
        "observations": np.concatenate(obs_buf),
        "actions": np.concatenate(act_buf),
        "terminals": np.concatenate(term_buf),
        "episode_ids": np.concatenate(ep_buf),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"\nwrote {out}")
    print(f"  transitions {len(data['observations'])}  episodes {kept}  "
          f"accept rate {kept / attempted:.0%}")
    print(f"  obs {data['observations'].shape}  act {data['actions'].shape}")
    print(f"  action range [{data['actions'].min():.2f}, {data['actions'].max():.2f}]")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--style", action="store_true",
                    help="add unobservable per-episode style offsets (breaks BC on purpose)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    collect(a.episodes, a.noise, a.seed, a.keep_failures, a.out, a.style)
    import os
    os._exit(0)
