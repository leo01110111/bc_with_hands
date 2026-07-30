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


class TokenPool(nn.Module):
    """Frozen DINOv3 tokens -> one conditioning vector.

    Three readouts are concatenated: spatial-softmax keypoints (where things
    are), attention-pooled content (what they are) and CLS (global summary).

    Attributes:
        out_dim: width of the vector concatenated onto proprioception
        num_queries: learned queries attending over the patch grid
        num_keypoints: spatial-softmax channels per camera
    """

    out_dim: int = 128
    num_queries: int = 4
    num_keypoints: int = 16

    @nn.compact
    def __call__(self, tokens):
        """
        tokens (B, n_cams, 1 + P, D)   index 0 of each camera is CLS
        ->     (B, out_dim)
        """
        b, n_cams, t, d = tokens.shape
        p = t - 1
        g = int(round(p ** 0.5))
        tokens = nn.LayerNorm()(tokens)
        cls, patches = tokens[:, :, 0], tokens[:, :, 1:]

        # Expected patch coordinate per channel. Attention pooling alone returns a
        # weighted average of patch *content*, which carries no position, so
        # without this the grid's spatial layout is thrown away.
        heat = nn.softmax(nn.Dense(self.num_keypoints)(patches), axis=2)
        coord = jnp.linspace(-1.0, 1.0, g)
        gy, gx = (a.reshape(-1) for a in jnp.meshgrid(coord, coord, indexing='ij'))
        keypoints = jnp.stack([jnp.einsum('bcpk,p->bck', heat, gx),
                               jnp.einsum('bcpk,p->bck', heat, gy)], -1).reshape(b, -1)

        pos = self.param('pos_embed', nn.initializers.normal(0.02), (1, n_cams, p, d))
        x = (patches + pos).reshape(b, n_cams * p, d)
        q = self.param('query', nn.initializers.normal(0.02), (self.num_queries, d))
        attn = nn.softmax(jnp.einsum('qd,bnd->bqn', q, x) / jnp.sqrt(d), axis=-1)
        pooled = jnp.einsum('bqn,bnd->bqd', attn, x).reshape(b, self.num_queries * d)

        h = jnp.concat([keypoints, pooled, cls.reshape(b, n_cams * d)], axis=-1)
        return nn.LayerNorm()(nn.Dense(self.out_dim)(h))


class VisionPolicy(nn.Module):
    """Velocity field conditioned on pooled image tokens plus proprioception.

    `observations` is a dict of {'tokens', 'proprio'} so it stays one pytree
    argument and flow.py needs no changes.
    """

    action_dim: int
    arch: str = 'adaln'
    vision_dim: int = 128

    @nn.compact
    def __call__(self, observations, x_t, t):
        v = TokenPool(out_dim=self.vision_dim)(observations['tokens'])
        obs = jnp.concat([observations['proprio'], v], axis=-1)
        return ARCHS[self.arch](self.action_dim)(obs, x_t, t)


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

