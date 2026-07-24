"""Render the scripted expert doing its thing -- `make video`."""

import numpy as np

from wuji_hands.expert import ScriptedExpert
from wuji_hands.leap_lift import EPISODE_STEPS, LeapLiftEnv

OUT = "/home/leo/wuji-hands/expert.mp4"


def main(n_episodes: int = 3, noise: float = 0.35, seed: int = 5, out: str = OUT) -> str:
    import imageio

    env = LeapLiftEnv(seed=seed)
    expert = ScriptedExpert(env, noise=noise)
    rng = np.random.default_rng(seed)

    frames, successes = [], 0
    for ep in range(n_episodes):
        env.reset(seed=seed * 100 + ep)
        expert.reset(rng)
        info = {"success": False}
        for t in range(EPISODE_STEPS):
            _, _, done, info = env.step(expert.act(t))
            if t % 2 == 0:
                frames.append(env.render(480, 352))
        successes += info["success"]
    env.close()

    imageio.mimsave(out, frames, fps=25)
    print(f"wrote {out}  ({successes}/{n_episodes} succeeded)")
    return out


if __name__ == "__main__":
    main()
    import os
    os._exit(0)
