"""L01 + L02 -- getting demonstrations into the model.

Fill in the TODOs. `make check` tells you when each one lands.
"""

import numpy as np


def load_demos(path: str) -> dict:
    """Load the .npz demo file into a dict of arrays.

    Returns a dict with at least:
        observations (N, 32) float32
        actions      (N, 20) float32
        terminals    (N,)    bool
        episode_ids  (N,)    int32
    """
    np.open("data/leap")
    # TODO(L01)
    return None


def sample_batch(dataset: dict, rng, batch_size: int) -> dict:
    """Draw `batch_size` random transitions.

    `rng` is a jax PRNGKey. Every array in the batch must be indexed with the
    SAME index vector, or your observations and actions will not correspond.
    """
    # TODO(L01)
    return None


class Normalizer:
    """Per-dimension mean/std normalisation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x) -> "Normalizer":
        """Compute per-dimension statistics from x of shape (N, D)."""
        # TODO(L02) -- careful: some obs dims never vary
        return None

    def normalize(self, x):
        # TODO(L02)
        raise NotImplementedError

    def denormalize(self, z):
        # TODO(L02)
        raise NotImplementedError
