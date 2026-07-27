"""L06 + L08 -- the training loop."""
import jax
import optax
from flax.training import train_state
from .nets import VelocityMLP
from .flow import flow_bc_loss
from .data import load_demos, Normalizer, sample_batch
import jax.numpy as jnp 

CKPT_DIR = "/home/leo/wuji_bc/checkpoints"


def create_train_state(rng, obs_dim: int, act_dim: int, lr: float = 3e-4):
    """Build the network, its parameters, and an Adam optimiser."""
    key1, key2 = jax.random.split(rng)
    model = VelocityMLP(act_dim)
    variables = model.init(key1, jax.random.normal(key2, (1, obs_dim)), jax.random.normal(key2, (1, act_dim)), jnp.array([[1]]))
    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn = model.apply,
        params=variables,
        tx=tx)
    return state

@jax.jit
def update_step(state, batch, rng):
    """One gradient step. Returns (new_state, info_dict_with_'loss')."""
    loss, grad = jax.value_and_grad(flow_bc_loss)(state.params, state.apply_fn, batch, rng)
    state = state.apply_gradients(grads=grad)
    return (state, {'loss': loss})


def train(steps: int = 40_000, batch_size: int = 256, seed: int = 0, **kwargs) -> dict:
    """Train on the real demos and save the params.

    Returns {'state': ..., 'val_loss': float}.

    Hold out whole EPISODES for validation, not random transitions. Save the
    normaliser statistics next to the params -- level 9 needs them.
    """
    demos = load_demos("data/leap_lift_demos.npz")

    episode_ids = demos['episode_ids']
    unique_eps = jnp.unique(episode_ids)
    rng = jax.random.key(seed)
    rng, key1 = jax.random.split(rng)

    unique_eps = jax.random.permutation(key1, unique_eps)

    n_val = max(1, int(0.1 * unique_eps.shape[0]))
    val_eps, train_eps = unique_eps[:n_val], unique_eps[n_val:]
    train_mask = jnp.isin(episode_ids, train_eps)
    val_mask = jnp.isin(episode_ids, val_eps)
    train_set = {k: v[train_mask] for k, v in demos.items()}
    val_set = {k: v[val_mask] for k, v in demos.items()}

    obs_normalizer = Normalizer.fit(train_set["observations"])
    train_set['observations'] = obs_normalizer.normalize(train_set['observations'])
    val_set['observations'] = obs_normalizer.normalize(val_set['observations'])

    obs_dim = train_set['observations'].shape[-1]
    act_dim = train_set['actions'].shape[-1]
    rng, key2 = jax.random.split(rng)
    state = create_train_state(key2, obs_dim=obs_dim, act_dim=act_dim, lr=3e-4)
    final_loss = None
    for step in range(steps):
        rng, batch_key, step_key = jax.random.split(rng, 3)
        batch = sample_batch(train_set, batch_key, batch_size)
        state, info = update_step(state, batch, step_key)
        if step % 100 == 0:
            print(f"Train Loss {info['loss']}")
            #eval function
        final_loss = info['loss']

    return {'state': state, 'loss': final_loss}
