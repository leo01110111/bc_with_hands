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
    hidden_dims: tuple = (512, 512, 512, 512)

    @nn.compact
    def __call__(self, observations, x_t, t):
        """
        observations (B, obs_dim)
        x_t          (B, action_dim)   current point along the flow
        t            (B, 1)            time in [0, 1]
        ->           (B, action_dim)   velocity
        """
        input = jnp.concat([observations, x_t, t], axis=-1)
        x = nn.Dense(self.hidden_dims[0])(input)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dims[1])(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dims[2])(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dims[3])(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


def time_features(t, num_freqs: int = 32, max_freq: float = 100.0):
    """(B, 1) time -> (B, 2*num_freqs) sin/cos features.

    Frequencies are log-spaced in [1, max_freq]. The velocity field is smooth in
    t, so max_freq stays far below the ~10k used for token positions.
    """
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(max_freq), num_freqs))
    ang = t * freqs
    return jnp.concat([jnp.sin(ang), jnp.cos(ang)], axis=-1)


class AdaLNVelocityMLP(nn.Module):
    """Same signature as VelocityMLP, but t conditions every block via adaLN-Zero.

    obs and x_t go through the trunk; t only enters as per-block (scale, shift,
    gate) on a LayerNorm whose own affine is switched off. The modulation and
    output projections are zero-init, so at step 0 every block is the identity
    and the net outputs zero velocity.

    Attributes:
        action_dim: size of the action vector (20 for LeapLift, or H*20 chunked)
        width: trunk width
        num_blocks: number of residual adaLN blocks
        mlp_ratio: inner width of each block, as a multiple of `width`
        time_dim: width of the time embedding
        num_freqs / max_freq: Fourier features fed to the time embedding
    """

    action_dim: int
    width: int = 512
    num_blocks: int = 4
    mlp_ratio: int = 2
    time_dim: int = 128
    num_freqs: int = 32
    max_freq: float = 100.0

    @nn.compact
    def __call__(self, observations, x_t, t):
        """
        observations (B, obs_dim)
        x_t          (B, action_dim)   current point along the flow
        t            (B, 1)            time in [0, 1]
        ->           (B, action_dim)   velocity
        """
        zeros = nn.initializers.zeros

        c = nn.Dense(self.time_dim)(time_features(t, self.num_freqs, self.max_freq))
        c = nn.silu(c)
        c = nn.Dense(self.time_dim)(c)

        h = nn.Dense(self.width)(jnp.concat([observations, x_t], axis=-1))

        for _ in range(self.num_blocks):
            scale, shift, gate = jnp.split(
                nn.Dense(3 * self.width, kernel_init=zeros, bias_init=zeros)(nn.silu(c)),
                3, axis=-1)
            y = nn.LayerNorm(use_scale=False, use_bias=False)(h)
            y = y * (1 + scale) + shift
            y = nn.Dense(self.mlp_ratio * self.width)(y)
            y = nn.silu(y)
            y = nn.Dense(self.width)(y)
            h = h + gate * y

        scale, shift = jnp.split(
            nn.Dense(2 * self.width, kernel_init=zeros, bias_init=zeros)(nn.silu(c)),
            2, axis=-1)
        h = nn.LayerNorm(use_scale=False, use_bias=False)(h)
        h = h * (1 + scale) + shift
        return nn.Dense(self.action_dim, kernel_init=zeros)(h)


ARCHS = {
    'mlp': VelocityMLP,
    'adaln': AdaLNVelocityMLP,
}


def make_net(arch: str, action_dim: int):
    """Build a velocity field by name. Checkpoints store the name so rollout
    rebuilds the architecture the params actually came from."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, expected one of {sorted(ARCHS)}")
    return ARCHS[arch](action_dim)

