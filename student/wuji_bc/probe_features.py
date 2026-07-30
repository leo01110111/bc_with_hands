"""Can a linear readout of the frozen features locate the cube?

Ridge regression from DINOv3 tokens to cube_pos, scored on held-out episodes.
If this is poor, no amount of TokenPool tuning will rescue the policy -- the
information is not in the features.

    .venv-vision/bin/python -m wuji_bc.probe_features --n 4000
"""

import argparse
from pathlib import Path

import numpy as np

from wuji_bc.encoder import DINOV3

DATA = Path("/home/leo/wuji-hands/data")
CUBE_POS = slice(20, 23)
PROPRIO = np.r_[0:20, 30]


def ridge_r2(xtr, ytr, xte, yte, lam: float = 1.0) -> float:
    """R^2 of linear ridge, solved in sample space since D >> N."""
    xtr, xte = xtr.astype(np.float32), xte.astype(np.float32)
    mu, ymu = xtr.mean(0), ytr.mean(0)
    xtr, xte = xtr - mu, xte - mu

    k = xtr @ xtr.T
    alpha = np.linalg.solve(k + lam * np.eye(len(k), dtype=np.float32), ytr - ymu)
    pred = (xte @ xtr.T) @ alpha + ymu
    return 1.0 - ((yte - pred) ** 2).sum() / ((yte - yte.mean(0)) ** 2).sum()


def main(npz_path: Path, frames_path: Path, size: str, n: int, max_step: int) -> None:
    demos = np.load(npz_path)
    frames = np.load(frames_path, mmap_mode="r")
    eps = demos["episode_ids"]

    rng = np.random.default_rng(0)
    pool = np.arange(len(frames))
    if max_step:
        # After contact the cube sits at the grasp centre, so proprioception
        # alone predicts cube_pos almost perfectly. Only the approach phase
        # tests whether the features carry cube position.
        pool = pool[(pool - np.searchsorted(eps, eps)) < max_step]
    idx = np.sort(rng.choice(pool, min(n, len(pool)), replace=False))
    # Split on episodes, not transitions: neighbouring steps are near-duplicates
    # and would leak the answer across the split.
    val_eps = set(rng.choice(np.unique(eps), max(1, len(np.unique(eps)) // 5),
                             replace=False).tolist())
    is_val = np.array([e in val_eps for e in eps[idx]])

    y = np.asarray(demos["observations"][idx][:, CUBE_POS], dtype=np.float32)
    enc = DINOV3(size=size).load()

    tokens = []
    for cam in range(frames.shape[1]):
        step = 512
        out = [enc.infer(np.asarray(frames[idx[i:i + step], cam]))
               for i in range(0, len(idx), step)]
        tokens.append(np.concatenate(out))
    tokens = np.stack(tokens, axis=1)                       # (n, cams, 1+P, D)

    cls = tokens[:, :, 0].reshape(len(idx), -1)
    patches = tokens[:, :, 1:]
    grid = patches.reshape(len(idx), -1)
    pooled = patches.mean(2).reshape(len(idx), -1)
    proprio = np.asarray(demos["observations"][idx][:, PROPRIO], dtype=np.float32)

    print(f"\n{len(idx)} frames, {is_val.sum()} held out, target cube_pos (3)"
          f"{'' if not max_step else f'  [steps < {max_step} only]'}\n")
    for name, x in (("proprio only (floor)", proprio),
                    ("CLS only", cls),
                    ("patches mean-pooled", pooled),
                    ("full patch grid", grid)):
        r2 = ridge_r2(x[~is_val], y[~is_val], x[is_val], y[is_val])
        print(f"  {name:22s} dim {x.shape[1]:>7,}   R2 {r2:6.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=DATA / "leap_lift_pixels.npz")
    ap.add_argument("--frames", type=Path, default=DATA / "leap_lift_frames.npy")
    ap.add_argument("--size", default="s", choices=sorted(DINOV3.REPOS))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--max-step", type=int, default=0,
                    help="restrict to the first K steps of each episode (approach phase)")
    a = ap.parse_args()
    main(a.npz, a.frames, a.size, a.n, a.max_step)
