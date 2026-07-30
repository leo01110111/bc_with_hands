"""Encode the rendered frames once with the frozen backbone.

Writes (N, n_cams, 1 + num_patches, embed_dim) float16 for train.py to load.
Runs under .venv-vision, not the jax venv:

    .venv-vision/bin/python -m wuji_bc.cache_features --preview 4
    .venv-vision/bin/python -m wuji_bc.cache_features
"""

import argparse
import time
from pathlib import Path

import numpy as np

from wuji_bc.encoder import DINOV3

DATA = Path("/home/leo/wuji-hands/data")
DEFAULT_FRAMES = DATA / "leap_lift_frames.npy"


def preview(frames_path: Path, size: str, n: int, out: Path) -> None:
    """PCA the patch grid of a few frames to RGB and write a PNG.

    Working features look like a crude segmentation; noise means the
    preprocessing or the weights are wrong.
    """
    from PIL import Image

    frames = np.load(frames_path, mmap_mode="r")
    # Episodes are a fixed length, so evenly spaced indices all land on the same
    # phase of the motion and the preview shows near-duplicates.
    idx = np.sort(np.random.default_rng(0).choice(len(frames), n, replace=False))
    enc = DINOV3(size=size).load()

    rows = []
    for cam in range(frames.shape[1]):
        imgs = np.asarray(frames[idx, cam])
        patches = enc.infer(imgs)[:, 1:]
        mean = patches.reshape(-1, enc.embed_dim).mean(0)
        comps = np.linalg.svd(patches.reshape(-1, enc.embed_dim) - mean,
                              full_matrices=False)[2][:3]

        proj = (patches - mean) @ comps.T
        lo, hi = proj.min((0, 1)), proj.max((0, 1))
        proj = ((proj - lo) / (hi - lo) * 255).astype(np.uint8)

        g = int(round(enc.num_patches ** 0.5))
        pca = proj.reshape(len(idx), g, g, 3).repeat(16, 1).repeat(16, 2)
        rows.append(np.concatenate(list(imgs), axis=1))
        rows.append(np.concatenate(list(pca), axis=1))

    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print(f"wrote {out}  (rows alternate: frame, then its PCA'd patch grid)")


def cache(frames_path: Path, size: str, out: Path, batch: int) -> None:
    frames = np.load(frames_path, mmap_mode="r")
    n, n_cams = frames.shape[:2]
    enc = DINOV3(size=size, batch_size=batch).load()

    feats = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float16,
        shape=(n, n_cams, 1 + enc.num_patches, enc.embed_dim),
    )
    print(f"{n} steps x {n_cams} cams -> {feats.shape} "
          f"({feats.nbytes / 1e9:.1f} GB)", flush=True)

    t0 = time.time()
    step = max(1, batch // n_cams)
    for i in range(0, n, step):
        chunk = np.asarray(frames[i : i + step])
        b = len(chunk)
        tokens = enc.infer(chunk.reshape(b * n_cams, *chunk.shape[2:]))
        feats[i : i + b] = tokens.reshape(b, n_cams, *tokens.shape[1:]).astype(np.float16)
        if (i // step) % 20 == 0:
            done = i + b
            print(f"  {done}/{n}  ({done / max(time.time() - t0, 1e-9):.0f} steps/s)",
                  flush=True)

    feats.flush()
    print(f"\nwrote {out}  {feats.shape} float16  in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--size", default="s", choices=sorted(DINOV3.REPOS))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--preview", type=int, default=0,
                    help="PCA this many frames to a PNG and exit, without caching")
    a = ap.parse_args()

    if a.preview:
        preview(a.frames, a.size, a.preview, a.out or DATA / "dino_preview.png")
    else:
        cache(a.frames, a.size,
              a.out or DATA / f"leap_lift_dinov3_{a.size}.npy", a.batch)
