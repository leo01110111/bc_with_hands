"""Frozen vision backbones -- pixels to patch tokens.

torch is imported inside load() so this stays importable from the jax venv.
"""

import numpy as np

IMAGE_SIZE = 224
PATCH = 16

# DINOv3 inherits DINOv2's ImageNet preprocessing.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VisionEncoder:
    """Image -> per-patch feature grid. Stateful: call load() before infer()."""

    embed_dim: int
    num_patches: int

    def load(self) -> "VisionEncoder":
        raise NotImplementedError

    def infer(self, images: np.ndarray) -> np.ndarray:
        """(B, 224, 224, 3) uint8 -> (B, 1 + num_patches, embed_dim) float32.

        Index 0 is CLS; the rest is the patch grid, row-major, reshapeable to
        (14, 14, dim). Register tokens are dropped.
        """
        raise NotImplementedError


class DINOV3(VisionEncoder):
    """Frozen DINOv3 ViT via the HuggingFace PyTorch weights.

    The repo is license-gated -- accept the terms on the model page and
    `hf auth login` before this will download.

    Attributes:
        size: 's' (384-dim, 21M), 'b' (768-dim, 86M) or 'l' (1024-dim, 300M)
        device: keep this off the GPU jax has taken
        batch_size: inference chunk, bounded by memory not correctness
    """

    REPOS = {
        "s": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }

    def __init__(self, size: str = "s", device: str = "cuda", batch_size: int = 256):
        if size not in self.REPOS:
            raise ValueError(f"unknown size {size!r}, expected one of {sorted(self.REPOS)}")
        self.size = size
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def load(self) -> "DINOV3":
        import torch
        from transformers import AutoModel

        self._torch = torch
        self._model = AutoModel.from_pretrained(self.REPOS[self.size]).eval().to(self.device)

        cfg = self._model.config
        self.embed_dim = cfg.hidden_size
        self.num_patches = (IMAGE_SIZE // PATCH) ** 2
        self._n_registers = getattr(cfg, "num_register_tokens", 0)
        return self

    def infer(self, images: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call load() first")
        if images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(
                f"expected (B, {IMAGE_SIZE}, {IMAGE_SIZE}, 3) uint8, got {images.shape}"
            )
        torch = self._torch

        out = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size].astype(np.float32) / 255.0
            chunk = (chunk - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(self.device)
            with torch.inference_mode():
                tokens = self._model(pixel_values=x).last_hidden_state
            cls, patches = tokens[:, :1], tokens[:, 1 + self._n_registers :]
            out.append(torch.cat([cls, patches], dim=1).float().cpu().numpy())

        return np.concatenate(out)
