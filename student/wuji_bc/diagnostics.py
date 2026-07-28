"""L11 -- why flow matching rather than a plain regressor."""

import functools

import numpy as np

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

from .nets import VelocityMLP
from .flow import flow_bc_loss, sample_actions


class MSE_MLP(nn.Module):
    """Predicts an action directly from an observation -- no noise, no time.

    Attributes:
        action_dim: size of the action vector
        hidden_dims: MLP widths
    """

    action_dim: int
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, observations):
        x = nn.Dense(self.hidden_dims[0])(observations)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dims[1])(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


def mse_loss(params, apply_fn, batch, rng):
    """E || a_theta(obs) - a ||^2. Same signature as flow_bc_loss, so one
    update step drives both models."""
    del rng
    pred = apply_fn(params, batch['observations'])
    return jnp.mean((pred - batch['actions']) ** 2)


def create_train_state(rng, model, init_args, lr):
    """Build parameters and an Adam optimiser. `init_args` are example inputs
    matching that model's own __call__ signature."""
    variables = model.init(rng, *init_args)
    tx = optax.adam(lr)
    return train_state.TrainState.create(
        apply_fn=lambda params, *args: model.apply({'params': params}, *args),
        params=variables['params'],
        tx=tx)


@functools.partial(jax.jit, static_argnums=0)
def update_step(loss_fn, state, batch, rng):
    """One gradient step. Returns (new_state, info_dict_with_'loss')."""
    loss, grad = jax.value_and_grad(loss_fn)(state.params, state.apply_fn, batch, rng)
    return state.apply_gradients(grads=grad), {'loss': loss}


def train_model(loss_fn, state, batch, rng, steps):
    for _ in range(steps):
        rng, step_key = jax.random.split(rng)
        state, info = update_step(loss_fn, state, batch, step_key)
    return state, info['loss']


def fit_bimodal_toy(steps: int = 3000, n_samples: int = 256, flow_steps: int = 50,
                    lr: float = 1e-3, seed: int = 0) -> dict:
    """One observation, two equally-good expert actions (-1 and +1).

    Fit your flow model AND a deterministic MSE regressor on it.
    Return {'flow_samples': (N,), 'mse_samples': (N,)}.
    """
    act_dim = 1
    batch = {
        'observations': jnp.zeros((4, 1)),
        'actions': jnp.array([[1.0], [1.0], [-1.0], [-1.0]]),
    }

    rng = jax.random.key(seed)
    rng, mse_key, flow_key, sample_key = jax.random.split(rng, 4)

    mse_init, mse_train = jax.random.split(mse_key)
    mse_state = create_train_state(
        mse_init, MSE_MLP(act_dim), (jnp.zeros((1, 1)),), lr)
    mse_state, mse_final = train_model(mse_loss, mse_state, batch, mse_train, steps)

    flow_init, flow_train = jax.random.split(flow_key)
    flow_state = create_train_state(
        flow_init, VelocityMLP(act_dim),
        (jnp.zeros((1, 1)), jnp.zeros((1, act_dim)), jnp.zeros((1, 1))), lr)
    flow_state, flow_final = train_model(flow_bc_loss, flow_state, batch, flow_train, steps)

    eval_obs = jnp.zeros((n_samples, 1))
    flow_samples = sample_actions(flow_state.params, flow_state.apply_fn, eval_obs,
                                  sample_key, flow_steps, act_dim).reshape(-1)
    # Deterministic: one prediction, repeated so the two arrays compare directly.
    mse_pred = mse_state.apply_fn(mse_state.params, jnp.zeros((1, 1))).reshape(-1)
    mse_samples = jnp.repeat(mse_pred, n_samples)

    return {
        'flow_samples': np.asarray(flow_samples),
        'mse_samples': np.asarray(mse_samples),
        'flow_loss': float(flow_final),
        'mse_loss': float(mse_final),
    }


if __name__ == "__main__":
    out = fit_bimodal_toy()
    flow, mse = out['flow_samples'], out['mse_samples']
    print(f"mse   loss {out['mse_loss']:.4f}  prediction {mse[0]:+.3f}")
    print(f"flow loss {out['flow_loss']:.4f}  predictions {flow}")
