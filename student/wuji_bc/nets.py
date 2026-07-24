"""L03 -- the conditional velocity field v_theta(obs, x_t, t)."""

import flax.linen as nn
import jax.numpy as jnp


class VelocityMLP(nn.Module):
    """Predicts a velocity in ACTION space, conditioned on obs, x_t and t.

    Attributes:
        action_dim: size of the action vector (20 for LeapLift, or H*20 chunked)
        hidden_dims: MLP widths
    """

    action_dim: int
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, observations, x_t, t):
        """
        observations (B, obs_dim)
        x_t          (B, action_dim)   current point along the flow
        t            (B, 1)            time in [0, 1]
        ->           (B, action_dim)   velocity
        """
        # TODO(L03)
        return None
