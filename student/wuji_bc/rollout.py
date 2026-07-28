"""L09, L10 + L12 -- run the policy on the actual hand."""

import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.serialization as fs

from .nets import VelocityMLP
from .data import MinMaxNormalizer
from .flow import sample_actions

CKPT_DIR = Path("/home/leo/wuji-hands/checkpoints")
CKPT_FILE = CKPT_DIR / "ckpt.msgpack"

# collect() resets with `collect_seed * 100_000 + attempt`, so the demonstrated
# initial conditions are small non-negative ints (0..~1538 for our dataset).
# Evaluating at seed 1234 therefore replays demo episode 1233 EXACTLY. Offsetting
# every rollout past that range keeps eval placements genuinely unseen. Negative
# seeds would be the obvious fix but numpy's default_rng rejects them; this
# constant is not a multiple of 100_000, so no plausible collect_seed reaches it.
EVAL_SEED_BASE = 987_654_321

def load_policy(checkpoint_file=CKPT_FILE):
    """Params + apply_fn + the TRAINING normalizer, straight off disk.

    `act_dim` is what the NET predicts (H * env action dim); `horizon` is H.
    """
    ckpt = fs.msgpack_restore(Path(checkpoint_file).read_bytes())

    obs_norm = MinMaxNormalizer(jnp.asarray(ckpt["obs_min"]), jnp.asarray(ckpt["obs_max"]))
    act_dim = int(ckpt["act_dim"])
    horizon = int(ckpt.get("horizon", 1))     # checkpoints predating chunking
    params = ckpt["params"]

    model = VelocityMLP(act_dim)

    def apply_fn(p, obs, x_t, t):
        return model.apply({"params": p}, obs, x_t, t)

    return params, apply_fn, obs_norm, act_dim, horizon

def evaluate_policy(n_episodes: int = 20, seed: int = 0, flow_steps: int = 10,
                    video_path: str | None = None, checkpoint_file=CKPT_FILE,
                    deterministic: bool = True, chunk_mode: str = "open_loop") -> dict:
    """Load trained params, roll out in LeapLiftEnv, return {'success_rate': ...}.

        from wuji_hands.leap_lift import LeapLiftEnv, EPISODE_STEPS

    Pass video_path to save a rollout -- watching the failures is the fastest
    debugging tool you have here.

    Config:
        deterministic  x_0 = 0 (default) vs a fresh N(0, I) draw per query.
        chunk_mode     for a horizon>1 checkpoint: 'open_loop' executes all H
                       actions before re-querying, 'receding' re-queries every
                       step and executes only the first. Ignored when H == 1,
                       where the two are identical.
    """
    from wuji_hands.leap_lift import LeapLiftEnv, CONTROL_DT

    params, apply_fn, obs_norm, act_dim, horizon = load_policy(checkpoint_file)
    env_act_dim = act_dim // horizon

    @jax.jit
    def act(params, obs, rng):
        # x_0 = 0 rather than a fresh draw. The residual dependence on x_0 is
        # model/discretisation error, not learned data noise (it survives
        # unchanged on noise-free demos), and sampling it turns a feedback law
        # into a random walk. Measured 15-40% -> 95% over 20 episodes.
        noises = jnp.zeros((obs.shape[0], act_dim)) if deterministic else None
        return sample_actions(params, apply_fn, obs, rng, flow_steps, act_dim, noises)

    env = LeapLiftEnv(seed=seed)
    rng = jax.random.key(seed)
    frames, successes = [], 0

    for ep in range(n_episodes):
        obs = env.reset(seed=EVAL_SEED_BASE + seed + ep)
        done, info = False, {}
        pending = []                     
        while not done:
            if not pending:
                rng, akey = jax.random.split(rng)
                o = obs_norm.normalize(jnp.asarray(obs)[None, :])
                chunk = np.asarray(act(params, o, akey))[0].reshape(horizon, env_act_dim)
                pending = list(chunk) if chunk_mode == "open_loop" else [chunk[0]]
            obs, _, done, info = env.step(pending.pop(0))
            if video_path is not None and ep == 0:
                frames.append(env.render())
        successes += bool(info["success"])

    if video_path is not None and frames:
        import imageio
        imageio.mimsave(video_path, frames, fps=int(round(1.0/CONTROL_DT)))

    env.close()

    return {"success_rate": successes/n_episodes,
            "n_episodes": n_episodes,
            "flow_steps": flow_steps,
            "horizon": horizon,
            "chunk_mode": chunk_mode if horizon > 1 else "n/a",
            "deterministic": deterministic}


def flow_steps_sweep(steps_list=(1, 2, 4, 10), n_episodes: int = 20, seed: int = 1234) -> dict:
    """Success rate as a function of the number of Euler steps at inference.

    Returns {flow_steps: success_rate}. Same trained params throughout -- the
    only thing changing is how finely you integrate the ODE.
    """
    
    return {steps: evaluate_policy(n_episodes, seed, steps)['success_rate']
            for steps in steps_list}
