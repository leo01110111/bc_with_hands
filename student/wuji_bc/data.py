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
    demos = np.load(path)
    return demos

import jax
import jax.numpy as jnp

def sample_batch(dataset: dict, rng, batch_size: int) -> dict:
    """Draw `batch_size` random transitions.

    `rng` is a jax PRNGKey. Every array in the batch must be indexed with the
    SAME index vector, or your observations and actions will not correspond.
    """
    N = next(iter(dataset.values())).shape[0] #the trajectory size
    random_sample = jax.random.randint(rng, (batch_size), 0, N)
    subset = dict()
    for key, _ in dataset.items():
        subset[key] = dataset[key][random_sample]

    return subset


class Normalizer:
    """Per-dimension mean/std normalisation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x) -> "Normalizer":
        """Compute per-dimension statistics from x of shape (N, D)."""
        mean = jnp.mean(x, axis=0)
        std = jnp.sqrt(jnp.var(x, axis=0))
        return Normalizer(mean, std)

    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, z):
        return self.std * z + self.mean


if __name__ == "__main__":
    load_demos("data/leap_lift_demos.npz")