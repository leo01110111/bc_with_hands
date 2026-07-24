"""L04, L05, L07 -- flow matching: targets, loss, and sampling."""

import jax
import jax.numpy as jnp


def get_flow_targets(actions, rng):
    """Sample a point along the noise -> action path.

        x_0 ~ N(0, I)
        t   ~ U(0, 1)          per sample, shape (B, 1)
        x_t = (1 - t) * x_0 + t * x_1
        v   = x_1 - x_0

    Returns (x_t, v_target, t).
    """
    # TODO(L04)
    return None


def flow_bc_loss(params, apply_fn, batch, rng):
    """E || v_theta(obs, x_t, t) - (x_1 - x_0) ||^2, as a scalar.

    `apply_fn(params, observations, x_t, t) -> velocity`
    Return either `loss` or `(loss, info_dict)`.
    """
    # TODO(L05)
    return None


def sample_actions(params, apply_fn, observations, rng, flow_steps, act_dim, noises=None):
    """Integrate the learned velocity field from noise to an action.

        x <- noises, or N(0, I) if noises is None
        for i in range(flow_steps):
            t = i / flow_steps                (shape (B, 1))
            x <- x + apply_fn(params, obs, x, t) * (1 / flow_steps)
        return clip(x, -1, 1)
    """
    # TODO(L07)
    return None
