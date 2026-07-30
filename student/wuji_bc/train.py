"""L06 + L08 -- the training loop."""
import jax
import optax
import flax.serialization as fs
from flax.training import train_state
from .nets import make_net, VisionPolicy
from .flow import flow_bc_loss
from .data import (load_demos, MinMaxNormalizer, sample_batch, make_action_chunks,
                   load_pixel_demos, sample_pixel_batch)
import jax.numpy as jnp
import numpy as np
from pathlib import Path

CKPT_DIR = Path("/home/leo/wuji-hands/checkpoints")
CKPT_FILE = CKPT_DIR / "ckpt.msgpack"


def create_train_state(rng, obs_dim: int, act_dim: int, lr: float = 3e-4, arch: str = "mlp"):
    """Build the network, its parameters, and an Adam optimiser."""
    key1, key2 = jax.random.split(rng)
    model = make_net(arch, act_dim)
    variables = model.init(key1, jax.random.normal(key2, (1, obs_dim)), jax.random.normal(key2, (1, act_dim)), jnp.array([[1]]))
    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn = lambda params, *args: model.apply({'params': params}, *args),
        params=variables['params'],
        tx=tx)
    return state

def create_vision_train_state(rng, proprio_dim: int, act_dim: int, token_shape,
                              lr: float = 3e-4, arch: str = "adaln", vision_dim: int = 128):
    """Train state for VisionPolicy. token_shape is (n_cams, 1 + n_patches, embed_dim)."""
    key1, key2 = jax.random.split(rng)
    model = VisionPolicy(action_dim=act_dim, arch=arch, vision_dim=vision_dim)
    obs = {'proprio': jax.random.normal(key2, (1, proprio_dim)),
           'tokens': jax.random.normal(key2, (1,) + tuple(token_shape))}
    variables = model.init(key1, obs, jax.random.normal(key2, (1, act_dim)), jnp.zeros((1, 1)))
    return train_state.TrainState.create(
        apply_fn=lambda params, *args: model.apply({'params': params}, *args),
        params=variables['params'],
        tx=optax.adam(lr))


@jax.jit
def update_step(state, batch, rng):
    """One gradient step. Returns (new_state, info_dict_with_'loss')."""
    loss, grad = jax.value_and_grad(flow_bc_loss)(state.params, state.apply_fn, batch, rng)
    state = state.apply_gradients(grads=grad)
    return (state, {'loss': loss})

@jax.jit
def eval_step(state, batch, rng):
    loss = flow_bc_loss(state.params, state.apply_fn, batch, rng)
    return loss

def train(steps: int = 40_000, eval_interval: int = 1000, batch_size: int = 256, seed: int = 0,
          horizon: int = 1, demos_path: str = "data/leap_lift_demos.npz", arch: str = "mlp",
          ckpt_file=None, wandb_project: str | None = None, wandb_run_name: str | None = None,
          log_interval: int = 100, **kwargs) -> dict:
    """Train on the real demos and save the params.

    Returns {'state': ..., 'final_train_loss': float, 'final_val_loss': float}.

    Hold out whole EPISODES for validation, not random transitions. Save the
    normaliser statistics next to the params -- level 9 needs them.

    `horizon` > 1 turns on action chunking: the net predicts the next H actions
    as one H*act_dim vector. horizon=1 (the default) is the single-step policy
    and is completely unaffected by the chunking path.

    `arch` picks the velocity field: 'mlp' (concat obs, x_t, t) or 'adaln'
    (t enters every block as adaLN-Zero modulation). It is saved in the
    checkpoint so rollout rebuilds the matching network.

    Pass `wandb_project` to log losses and upload the checkpoint as an artifact;
    with it unset nothing touches the network.
    """
    run = None
    if wandb_project is not None:
        import wandb
        run = wandb.init(project=wandb_project, name=wandb_run_name, config={
            'steps': steps, 'batch_size': batch_size, 'seed': seed, 'horizon': horizon,
            'lr': 3e-4, 'demos_path': demos_path, 'eval_interval': eval_interval, 'arch': arch,
        })

    demos = load_demos(demos_path)

    episode_ids = demos['episode_ids']
    # Chunk before the split so a chunk never spans two episodes.
    demos['actions'] = make_action_chunks(demos['actions'], episode_ids, horizon)
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

    obs_normalizer = MinMaxNormalizer.fit(train_set["observations"])
    train_set['observations'] = obs_normalizer.normalize(train_set['observations'])
    val_set['observations'] = obs_normalizer.normalize(val_set['observations'])

    obs_dim = train_set['observations'].shape[-1]
    act_dim = train_set['actions'].shape[-1]
    rng, key2 = jax.random.split(rng)
    state = create_train_state(key2, obs_dim=obs_dim, act_dim=act_dim, lr=3e-4, arch=arch)

    rng, val_key, val_batch_key = jax.random.split(rng, 3)
    val_batch = sample_batch(val_set, val_batch_key, batch_size)
    final_train_loss = None
    val_loss = None

    for step in range(steps):
        rng, batch_key, step_key = jax.random.split(rng, 3)

        batch = sample_batch(train_set, batch_key, batch_size)
        
        state, info = update_step(state, batch, step_key)
        
        if step % eval_interval == 0:
            val_loss = eval_step(state, val_batch, val_key)
            print(f"step {step}  train {info['loss']:.4f}  val {val_loss:.4f}")
            if run is not None:
                run.log({'val_loss': float(val_loss)}, step=step)

        # float() blocks until the step is done, so only do it when logging.
        if run is not None and step % log_interval == 0:
            run.log({'train_loss': float(info['loss'])}, step=step)

        final_train_loss = info['loss']

    val_loss = eval_step(state, val_batch, val_key)
    print(f"step {step}  train {final_train_loss:.4f}  val {val_loss:.4f}")
    if run is not None:
        run.log({'train_loss': float(final_train_loss), 'val_loss': float(val_loss)}, step=step)

    # Params and the normaliser go in ONE file: written together, they cannot
    # drift apart the way two files can (params from run B, stats from run A
    # still have matching shapes and just quietly perform worse). L09 loads it.
    out = Path(ckpt_file) if ckpt_file is not None else CKPT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(fs.msgpack_serialize({
        'params': state.params,
        'obs_min': np.asarray(obs_normalizer.min),
        'obs_max': np.asarray(obs_normalizer.max),
        'obs_dim': np.int32(obs_dim),
        'act_dim': np.int32(act_dim),      # H * env action dim
        'horizon': np.int32(horizon),
        'arch': arch,
    }))

    if run is not None:
        artifact = wandb.Artifact(f"ckpt-{run.id}", type="model",
                                  metadata={'horizon': horizon, 'val_loss': float(val_loss)})
        artifact.add_file(str(out))
        run.log_artifact(artifact)
        run.finish()

    return {'state': state, 'train_loss': final_train_loss, 'val_loss': float(val_loss)}

def train_vision(steps: int = 40_000, eval_interval: int = 1000, batch_size: int = 128,
                 seed: int = 0, horizon: int = 1, arch: str = "adaln", vision_dim: int = 128,
                 proprio_only: bool = True, lr: float = 3e-4,
                 demos_path: str = "data/leap_lift_pixels.npz",
                 features_path: str = "data/leap_lift_dinov3_s.npy",
                 ckpt_file=None, wandb_project: str | None = None,
                 wandb_run_name: str | None = None, log_interval: int = 100) -> dict:
    """Same loop as train(), conditioned on cached DINOv3 tokens as well as state.

    Only TokenPool and the velocity field train; the backbone was frozen when
    the features were cached.
    """
    run = None
    if wandb_project is not None:
        import wandb
        run = wandb.init(project=wandb_project, name=wandb_run_name, config={
            'steps': steps, 'batch_size': batch_size, 'seed': seed, 'horizon': horizon,
            'lr': lr, 'arch': arch, 'vision_dim': vision_dim, 'proprio_only': proprio_only,
            'features_path': features_path,
        })

    demos = load_pixel_demos(demos_path, features_path, proprio_only=proprio_only)

    episode_ids = demos['episode_ids']
    demos['actions'] = make_action_chunks(demos['actions'], episode_ids, horizon)

    rng = jax.random.key(seed)
    rng, key1 = jax.random.split(rng)
    unique_eps = jax.random.permutation(key1, jnp.unique(episode_ids))
    n_val = max(1, int(0.1 * unique_eps.shape[0]))
    val_mask = jnp.isin(episode_ids, unique_eps[:n_val])
    train_set = {k: v[~val_mask] for k, v in demos.items()}
    val_set = {k: v[val_mask] for k, v in demos.items()}

    # Tokens are left alone -- TokenPool starts with a LayerNorm.
    obs_normalizer = MinMaxNormalizer.fit(train_set['observations'])
    for s in (train_set, val_set):
        s['observations'] = obs_normalizer.normalize(s['observations'])

    proprio_dim = train_set['observations'].shape[-1]
    act_dim = train_set['actions'].shape[-1]
    token_shape = train_set['tokens'].shape[1:]

    rng, key2 = jax.random.split(rng)
    state = create_vision_train_state(key2, proprio_dim, act_dim, token_shape,
                                      lr=lr, arch=arch, vision_dim=vision_dim)
    print(f"proprio {proprio_dim}  tokens {token_shape}  act {act_dim}  "
          f"params {sum(x.size for x in jax.tree.leaves(state.params)):,}", flush=True)

    rng, val_key, val_batch_key = jax.random.split(rng, 3)
    val_batch = sample_pixel_batch(val_set, val_batch_key, batch_size)
    final_train_loss = val_loss = None

    for step in range(steps):
        rng, batch_key, step_key = jax.random.split(rng, 3)
        batch = sample_pixel_batch(train_set, batch_key, batch_size)
        state, info = update_step(state, batch, step_key)

        if step % eval_interval == 0:
            val_loss = eval_step(state, val_batch, val_key)
            print(f"step {step}  train {info['loss']:.4f}  val {val_loss:.4f}", flush=True)
            if run is not None:
                run.log({'val_loss': float(val_loss)}, step=step)
        if run is not None and step % log_interval == 0:
            run.log({'train_loss': float(info['loss'])}, step=step)
        final_train_loss = info['loss']

    val_loss = eval_step(state, val_batch, val_key)
    print(f"step {steps - 1}  train {final_train_loss:.4f}  val {val_loss:.4f}", flush=True)

    out = Path(ckpt_file) if ckpt_file is not None else CKPT_DIR / "ckpt_vision.msgpack"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(fs.msgpack_serialize({
        'params': state.params,
        'obs_min': np.asarray(obs_normalizer.min),
        'obs_max': np.asarray(obs_normalizer.max),
        'proprio_dim': np.int32(proprio_dim),
        'act_dim': np.int32(act_dim),
        'horizon': np.int32(horizon),
        'arch': arch,
        'vision_dim': np.int32(vision_dim),
        'proprio_only': proprio_only,
        'token_shape': np.asarray(token_shape, dtype=np.int32),
    }))
    print(f"wrote {out}", flush=True)

    if run is not None:
        artifact = wandb.Artifact(f"ckpt-{run.id}", type="model",
                                  metadata={'horizon': horizon, 'val_loss': float(val_loss)})
        artifact.add_file(str(out))
        run.log_artifact(artifact)
        run.finish()

    return {'state': state, 'train_loss': final_train_loss, 'val_loss': float(val_loss)}


"""
train does the following:
- loads data from a path
- creates a train and val set
- normalize obs 
- creates a train state
- makes a val batch
- training loop:
    - samples batch
    - updatss the state with the batch
    - does an eval every interval
- returns state and normalization statistics 
"""