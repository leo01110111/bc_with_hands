"""L01 + L02 -- getting demonstrations into the model.

Fill in the TODOs. `make check` tells you when each one lands.
"""

import numpy as np


def load_demos(path: str) -> dict:
    """Load the .npz demo file into a dict of Jax arrays.

    Returns a dict with at least:
        observations (N, 32) float32
        actions      (N, 20) float32
        terminals    (N,)    bool
        episode_ids  (N,)    int32
    """
    demos = np.load(path)
    demos = {k: jnp.asarray(v) for k, v in demos.items()}
    return demos

import jax
import jax.numpy as jnp

def make_action_chunks(actions, episode_ids, horizon: int):
    """Stack the next `horizon` actions into one flat target of size H*act_dim.

    Chunks never cross an episode boundary -- a chunk that would run past the
    end of its episode repeats that episode's last action instead. horizon=1
    returns `actions` unchanged, so the single-step path is untouched.
    """
    if horizon == 1:
        return actions

    a = np.asarray(actions)
    eps = np.asarray(episode_ids)
    n, act_dim = a.shape

    idx = np.arange(n)[:, None] + np.arange(horizon)[None, :]
    idx = np.clip(idx, 0, n - 1)
    # -1 marks "past the end of this episode"; the running max then carries the
    # last in-episode index forward, which is the repeat-last-action padding.
    idx = np.where(eps[idx] == eps[:, None], idx, -1)
    idx = np.maximum.accumulate(idx, axis=1)

    return jnp.asarray(a[idx].reshape(n, horizon * act_dim))


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


class ZScoreNormalizer:
    """Per-dimension mean/std normalisation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x) -> "ZScoreNormalizer":
        """Compute per-dimension statistics from x of shape (N, D)."""
        mean = jnp.mean(x, axis=0)
        std = jnp.sqrt(jnp.var(x, axis=0))
        return ZScoreNormalizer(mean, std)

    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, z):
        return self.std * z + self.mean


class MinMaxNormalizer:
    """1st and 99th percentile normalization"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    @classmethod
    def fit(cls, x):
        min = jnp.percentile(x, 1, axis=0)
        max = jnp.percentile(x, 99, axis=0)
        if max - min is 0:
            raise Exception("max=min not allowed")
        return MinMaxNormalizer(min, max)

    def normalize(self, x):
        return 2*(x-self.min)/(self.max-self.min) - 1 
    
    def denormalize(self, z):
        return (z+1)*(self.max-self.min)/2+self.min

if __name__ == "__main__":
    load_demos("data/leap_lift_demos.npz")