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
    key1, key2 = jax.random.split(rng)
    
    x_0 = jax.random.normal(key1, actions.shape)
    
    t = jax.random.uniform(key2, (actions.shape[0], 1))
    
    x_t = (1 - t) * x_0 + t * actions

    v_target = actions - x_0
    
    return (x_t, v_target, t) 

def flow_bc_loss(params, apply_fn, batch, rng):
    """E || v_theta(obs, x_t, t) - (x_1 - x_0) ||^2, as a scalar.

    `apply_fn(params, observations, x_t, t) -> velocity`
    Return either `loss` or `(loss, info_dict)`.
    """
    actions = batch['actions']

    x_t, v_target, t = get_flow_targets(actions, rng)

    loss = jnp.mean((apply_fn(params, batch['observations'], x_t, t) - v_target)**2, axis=(0,1))
    
    return loss

if __name__ == "__main__":
    rng = jax.random.PRNGKey(1)
    rng, actions_key, obs_key = jax.random.split(rng, 3)

    params = None
    apply_fn = lambda params, observations, x_t, t: x_t

    batch = {
        'actions': jax.random.normal(actions_key, (2, 3)),
        'observations': jax.random.normal(obs_key, (2, 4)),
    }

    print(flow_bc_loss(params, apply_fn, batch, rng))
    
def sample_actions(params, apply_fn, observations, rng, flow_steps, act_dim, noises=None):
    """Integrate the learned velocity field from noise to an action.

        x <- noises, or N(0, I) if noises is None
        for i in range(flow_steps):
            t = i / flow_steps                (shape (B, 1))
            x <- x + apply_fn(params, obs, x, t) * (1 / flow_steps)
        return clip(x, -1, 1)
    """
    x = jax.random.normal(rng, (observations.shape[0], act_dim))
    x = noises if noises is not None else jax.random.normal(rng, (observations.shape[0], act_dim))
    for i in range(flow_steps):
        t = jnp.full((x.shape[0],1), i/flow_steps)
        x = x + apply_fn(params, observations, x, t) * (1 / flow_steps)
    return jnp.clip(x, -1, 1)
