"""L06 + L08 -- the training loop."""

import jax
import optax
from flax.training import train_state

CKPT_DIR = "/home/leo/wuji_bc/checkpoints"


def create_train_state(rng, obs_dim: int, act_dim: int, lr: float = 3e-4):
    """Build the network, its parameters, and an Adam optimiser."""
    # TODO(L06)
    return None


def update_step(state, batch, rng):
    """One gradient step. Returns (new_state, info_dict_with_'loss')."""
    # TODO(L06)  -- add @jax.jit once it works
    return None


def train(steps: int = 40_000, batch_size: int = 256, seed: int = 0, **kwargs) -> dict:
    """Train on the real demos and save the params.

    Returns {'state': ..., 'val_loss': float}.

    Hold out whole EPISODES for validation, not random transitions. Save the
    normaliser statistics next to the params -- level 9 needs them.
    """
    # TODO(L08)
    return None
