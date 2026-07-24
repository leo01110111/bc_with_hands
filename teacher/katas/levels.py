"""The level ladder.

Each level is a small, checkable claim about code *you* write. The checks are
deliberately picky about shapes and about mathematical identities that only hold
if the implementation is actually right -- a test that just calls your function
and shrugs would teach you nothing.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from katas.runner import KataFail, NotImplementedYet

# Where the student's code lives. The teacher repo (this one) never contains
# solutions; `student` here is a symlink to it, and STUDENT overrides both.
ROOT = Path(__file__).resolve().parents[2]
SRC = str((Path(os.environ.get("STUDENT", ROOT / "student")) / "wuji_bc").resolve())


@dataclass
class Level:
    num: int
    title: str
    file: str
    brief: str
    check: Callable[[], None]
    hints: list[str] = field(default_factory=list)
    ogpo_refs: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers used by the checks
# --------------------------------------------------------------------------

def _import(mod: str):
    import importlib
    try:
        m = importlib.import_module(f"wuji_bc.{mod}")
        importlib.reload(m)
        return m
    except ModuleNotFoundError as e:
        raise KataFail(f"cannot import wuji_bc.{mod}: {e}", hint=f"does {SRC}/{mod}.py exist?")


def _get(mod: str, name: str):
    m = _import(mod)
    fn = getattr(m, name, None)
    if fn is None:
        raise KataFail(f"wuji_bc.{mod}.{name} does not exist",
                       hint=f"define {name} in {SRC}/{mod}.py")
    return fn


def _check_shape(got, expected, what: str, hint: str | None = None):
    if tuple(np.shape(got)) != tuple(expected):
        raise KataFail(f"{what} has the wrong shape",
                       expected=f"array{tuple(expected)}",
                       got=f"array{tuple(np.shape(got))}", hint=hint)


def _stub_guard(value, what: str):
    if value is None:
        raise NotImplementedYet(what)


DEMOS = str(ROOT / "data" / "leap_lift_demos.npz")
OBS_DIM, ACT_DIM = 31, 20


# --------------------------------------------------------------------------
# L01 -- dataset
# --------------------------------------------------------------------------

def check_01():
    load = _get("data", "load_demos")
    ds = load(DEMOS)
    _stub_guard(ds, "load_demos")

    for key in ("observations", "actions"):
        if key not in ds:
            raise KataFail(f"dataset is missing '{key}'",
                           hint="return a dict with observations/actions/terminals")
    n = len(ds["observations"])
    if n < 10_000:
        raise KataFail("dataset is suspiciously small", expected=">= 10000 transitions", got=n)
    _check_shape(ds["observations"], (n, OBS_DIM), "observations")
    _check_shape(ds["actions"], (n, ACT_DIM), "actions")

    sample = _get("data", "sample_batch")
    import jax
    rng = jax.random.PRNGKey(0)
    b = sample(ds, rng, 256)
    _stub_guard(b, "sample_batch")
    _check_shape(b["observations"], (256, OBS_DIM), "batch observations")
    _check_shape(b["actions"], (256, ACT_DIM), "batch actions")

    # different key -> different rows. Catches "I ignored the rng".
    b2 = sample(ds, jax.random.PRNGKey(1), 256)
    if np.allclose(np.asarray(b["observations"]), np.asarray(b2["observations"])):
        raise KataFail("two different rng keys produced identical batches",
                       hint="are you actually using the rng to pick indices?")
    # obs and action must come from the SAME transition
    obs_all = np.asarray(ds["observations"])
    act_all = np.asarray(ds["actions"])
    idx = [int(np.argmin(np.abs(obs_all - np.asarray(b["observations"])[i]).sum(1)))
           for i in range(8)]
    for i, j in enumerate(idx):
        if not np.allclose(act_all[j], np.asarray(b["actions"])[i], atol=1e-5):
            raise KataFail("observations and actions in a batch are not aligned",
                           hint="index every array with the SAME index vector")


# --------------------------------------------------------------------------
# L02 -- normalisation
# --------------------------------------------------------------------------

def check_02():
    Normalizer = _get("data", "Normalizer")
    load = _get("data", "load_demos")
    ds = load(DEMOS)
    obs = np.asarray(ds["observations"])

    nrm = Normalizer.fit(obs)
    _stub_guard(nrm, "Normalizer.fit")
    z = np.asarray(nrm.normalize(obs))

    if not np.allclose(z.mean(0), 0, atol=1e-3):
        raise KataFail("normalised observations are not zero-mean",
                       expected="|mean| < 1e-3", got=float(np.abs(z.mean(0)).max()))
    if not np.allclose(z.std(0), 1, atol=1e-2):
        raise KataFail("normalised observations are not unit-variance",
                       expected="std ~ 1", got=f"[{z.std(0).min():.3f}, {z.std(0).max():.3f}]",
                       hint="some obs dims are nearly constant -- guard the divide with an eps")
    back = np.asarray(nrm.denormalize(z))
    if not np.allclose(back, obs, atol=1e-3):
        raise KataFail("denormalize(normalize(x)) != x",
                       got=float(np.abs(back - obs).max()))
    if not np.all(np.isfinite(z)):
        raise KataFail("normalised observations contain NaN/inf",
                       hint="a zero-variance dim divided by zero")


# --------------------------------------------------------------------------
# L03 -- the velocity network
# --------------------------------------------------------------------------

def check_03():
    import jax, jax.numpy as jnp
    VelocityMLP = _get("nets", "VelocityMLP")

    net = VelocityMLP(action_dim=ACT_DIM, hidden_dims=(256, 256))
    rng = jax.random.PRNGKey(0)
    obs = jnp.zeros((8, OBS_DIM))
    x_t = jnp.zeros((8, ACT_DIM))
    t = jnp.zeros((8, 1))

    params = net.init(rng, obs, x_t, t)
    v = net.apply(params, obs, x_t, t)
    _stub_guard(v, "VelocityMLP.__call__")
    _check_shape(v, (8, ACT_DIM), "predicted velocity",
                 hint="the network outputs a velocity in ACTION space")

    # t must actually change the output, or you built a t-blind model that can
    # never represent a flow.
    v0 = net.apply(params, obs, x_t, jnp.zeros((8, 1)))
    v1 = net.apply(params, obs, x_t, jnp.ones((8, 1)))
    if jnp.allclose(v0, v1, atol=1e-6):
        raise KataFail("output does not depend on t",
                       hint="t must be fed into the network, not just accepted as an argument")

    # obs must matter too
    o1 = net.apply(params, jnp.ones((8, OBS_DIM)), x_t, t)
    if jnp.allclose(v0, o1, atol=1e-6):
        raise KataFail("output does not depend on the observation",
                       hint="this is a *conditional* flow -- condition on obs")

    # and x_t
    x1 = net.apply(params, obs, jnp.ones((8, ACT_DIM)), t)
    if jnp.allclose(v0, x1, atol=1e-6):
        raise KataFail("output does not depend on x_t",
                       hint="the velocity field is a function of the current point x_t")

    # batch independence: row i must not leak into row j
    big = net.apply(params, jnp.zeros((4, OBS_DIM)), jnp.zeros((4, ACT_DIM)), jnp.zeros((4, 1)))
    one = net.apply(params, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACT_DIM)), jnp.zeros((1, 1)))
    if not jnp.allclose(big[0], one[0], atol=1e-5):
        raise KataFail("per-sample outputs change with batch size",
                       hint="something is mixing across the batch axis (a mean? a BatchNorm?)")


# --------------------------------------------------------------------------
# L04 -- flow targets
# --------------------------------------------------------------------------

def check_04():
    import jax, jax.numpy as jnp
    get_flow_targets = _get("flow", "get_flow_targets")

    rng = jax.random.PRNGKey(0)
    actions = jax.random.uniform(jax.random.PRNGKey(9), (512, ACT_DIM), minval=-1, maxval=1)
    out = get_flow_targets(actions, rng)
    _stub_guard(out, "get_flow_targets")
    if not isinstance(out, tuple) or len(out) != 3:
        raise KataFail("get_flow_targets must return (x_t, v_target, t)", got=type(out).__name__)
    x_t, v_target, t = out

    _check_shape(t, (512, 1), "t",
                 hint="t is per-sample and must broadcast over the action dim -- (B, 1), not (B,)")
    _check_shape(x_t, (512, ACT_DIM), "x_t")
    _check_shape(v_target, (512, ACT_DIM), "v_target")

    t_np = np.asarray(t)
    if t_np.min() < 0 or t_np.max() > 1:
        raise KataFail("t must lie in [0, 1]", got=f"[{t_np.min():.3f}, {t_np.max():.3f}]")
    if t_np.std() < 0.05:
        raise KataFail("t is not being sampled -- it barely varies",
                       expected="t ~ Uniform(0,1), std ~ 0.29", got=f"std {t_np.std():.4f}")
    if np.allclose(t_np, t_np[0]):
        raise KataFail("every sample in the batch got the SAME t",
                       hint="sample t with shape (batch, 1), not shape (1, 1)")

    # The defining identity: x_t = (1-t) x_0 + t x_1 and v = x_1 - x_0 must be
    # consistent, i.e. x_t + (1-t) v == x_1 exactly.
    lhs = np.asarray(x_t) + (1 - t_np) * np.asarray(v_target)
    if not np.allclose(lhs, np.asarray(actions), atol=1e-4):
        raise KataFail("x_t, v_target and t are mutually inconsistent",
                       expected="x_t + (1-t)*v_target == actions",
                       got=f"max error {np.abs(lhs - np.asarray(actions)).max():.4f}",
                       hint="x_t=(1-t)*x_0+t*x_1 and v=x_1-x_0. Both use the SAME x_0.")

    # x_0 recovered from the identity should look like standard normal noise
    x0 = np.asarray(actions) - np.asarray(v_target)
    if abs(x0.std() - 1.0) > 0.15 or abs(x0.mean()) > 0.1:
        raise KataFail("x_0 does not look like N(0, I)",
                       expected="mean ~ 0, std ~ 1",
                       got=f"mean {x0.mean():.3f}, std {x0.std():.3f}",
                       hint="x_0 must be sampled from a standard normal")

    # rng must be respected
    x_t2, _, t2 = get_flow_targets(actions, jax.random.PRNGKey(1))
    if np.allclose(np.asarray(t2), t_np):
        raise KataFail("the rng argument is ignored -- same t for different keys")


# --------------------------------------------------------------------------
# L05 -- the loss
# --------------------------------------------------------------------------

def check_05():
    import jax, jax.numpy as jnp
    flow_bc_loss = _get("flow", "flow_bc_loss")
    VelocityMLP = _get("nets", "VelocityMLP")

    net = VelocityMLP(action_dim=ACT_DIM, hidden_dims=(64, 64))
    rng = jax.random.PRNGKey(0)
    obs = jax.random.normal(jax.random.PRNGKey(1), (128, OBS_DIM))
    act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(2), (128, ACT_DIM)))
    params = net.init(rng, obs, act, jnp.zeros((128, 1)))
    batch = {"observations": obs, "actions": act}

    loss = flow_bc_loss(params, net.apply, batch, rng)
    _stub_guard(loss, "flow_bc_loss")
    if isinstance(loss, tuple):
        loss = loss[0]
    if np.shape(loss) != ():
        raise KataFail("the loss must be a scalar", expected="shape ()",
                       got=f"shape {np.shape(loss)}",
                       hint="you need a mean over both the batch and the action dims")
    if not np.isfinite(float(loss)):
        raise KataFail("loss is NaN/inf at init")

    # An untrained net predicts ~0, so the loss should be about E||v||^2 = E||x_1-x_0||^2
    # = E||x_1||^2 + action_dim (since x_0 is independent standard normal).
    approx = float(jnp.mean(jnp.sum(act ** 2, axis=-1)) + ACT_DIM) / ACT_DIM
    got = float(loss)
    if not (0.25 * approx < got < 4 * approx):
        raise KataFail("loss magnitude is implausible for an untrained network",
                       expected=f"~{approx:.2f} (mean-squared velocity)", got=f"{got:.2f}",
                       hint="are you summing where you meant to average, or vice versa?")

    # A perfect model must score exactly zero.
    def perfect(_params, o, x, t, x1=act):
        return x1 - (x - t * x1) / jnp.maximum(1 - t, 1e-6)

    zero = flow_bc_loss(params, perfect, batch, rng)
    if isinstance(zero, tuple):
        zero = zero[0]
    if float(zero) > 1e-3:
        raise KataFail("a model that predicts v_target exactly does not get ~0 loss",
                       expected="~0", got=f"{float(zero):.4f}",
                       hint="check the sign and the argument order in your MSE")

    # gradients must flow
    g = jax.grad(lambda p: flow_bc_loss(p, net.apply, batch, rng)[0]
                 if isinstance(flow_bc_loss(p, net.apply, batch, rng), tuple)
                 else flow_bc_loss(p, net.apply, batch, rng))(params)
    flat = np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(g)])
    if np.abs(flat).max() < 1e-9:
        raise KataFail("gradients are all zero",
                       hint="did a stop_gradient or a detached constant sneak in?")


# --------------------------------------------------------------------------
# L06 -- one training step
# --------------------------------------------------------------------------

def check_06():
    import jax, jax.numpy as jnp
    create_train_state = _get("train", "create_train_state")
    update_step = _get("train", "update_step")

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, obs_dim=OBS_DIM, act_dim=ACT_DIM, lr=1e-3)
    _stub_guard(state, "create_train_state")

    obs = jax.random.normal(jax.random.PRNGKey(1), (256, OBS_DIM))
    act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(2), (256, ACT_DIM)))
    batch = {"observations": obs, "actions": act}

    out = update_step(state, batch, rng)
    _stub_guard(out, "update_step")
    if not isinstance(out, tuple) or len(out) != 2:
        raise KataFail("update_step must return (new_state, info_dict)", got=type(out).__name__)

    # Overfit one fixed batch: the loss has to come down a lot.
    losses = []
    s = state
    for i in range(400):
        s, info = update_step(s, batch, jax.random.fold_in(rng, i))
        losses.append(float(info["loss"] if isinstance(info, dict) else info))

    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    if not np.isfinite(last):
        raise KataFail("loss went to NaN during training", got=losses[-1])
    if last >= first * 0.7:
        raise KataFail("loss did not decrease on a fixed batch",
                       expected=f"< {first * 0.7:.3f} after 400 steps", got=f"{last:.3f}",
                       hint="is the optimiser actually applied? are you returning the NEW state?")
    if s is state:
        raise KataFail("update_step returned the same state object",
                       hint="flax TrainStates are immutable -- return the updated one")


# --------------------------------------------------------------------------
# L07 -- sampling (the ODE)
# --------------------------------------------------------------------------

def check_07():
    import jax, jax.numpy as jnp
    sample_actions = _get("flow", "sample_actions")

    # A velocity field that is constant in t and x: v(o, x, t) = c.
    # Euler-integrating it from x_0 over t in [0,1] must land exactly at x_0 + c.
    c = jnp.linspace(-0.5, 0.5, ACT_DIM)[None, :]

    def const_field(_params, o, x, t):
        return jnp.broadcast_to(c, x.shape)

    rng = jax.random.PRNGKey(0)
    obs = jnp.zeros((16, OBS_DIM))
    noise = jax.random.normal(jax.random.PRNGKey(5), (16, ACT_DIM))

    a = sample_actions(None, const_field, obs, rng, flow_steps=10, act_dim=ACT_DIM, noises=noise)
    _stub_guard(a, "sample_actions")
    _check_shape(a, (16, ACT_DIM), "sampled actions")

    expect = np.clip(np.asarray(noise) + np.asarray(c), -1, 1)
    if not np.allclose(np.asarray(a), expect, atol=1e-4):
        raise KataFail("Euler integration of a constant field is wrong",
                       expected="x_0 + c (then clipped to [-1,1])",
                       got=f"max error {np.abs(np.asarray(a) - expect).max():.4f}",
                       hint="dt = 1/flow_steps, and you take exactly flow_steps of them")

    # Step count must matter for a non-constant field...
    def lin_field(_params, o, x, t):
        return x * 0.5 + t

    a4 = sample_actions(None, lin_field, obs, rng, flow_steps=4, act_dim=ACT_DIM, noises=noise)
    a64 = sample_actions(None, lin_field, obs, rng, flow_steps=64, act_dim=ACT_DIM, noises=noise)
    if np.allclose(np.asarray(a4), np.asarray(a64), atol=1e-6):
        raise KataFail("flow_steps has no effect",
                       hint="are you looping flow_steps times, or just taking one step?")

    # ...and t must be passed in as (B, 1) marching 0 -> 1
    seen = []

    def record_t(_params, o, x, t):
        seen.append(np.asarray(t).copy())
        return jnp.zeros_like(x)

    sample_actions(None, record_t, obs, rng, flow_steps=5, act_dim=ACT_DIM, noises=noise)
    if len(seen) != 5:
        raise KataFail("the velocity field was not called flow_steps times",
                       expected=5, got=len(seen))
    if seen[0].shape != (16, 1):
        raise KataFail("t passed to the network has the wrong shape",
                       expected="(16, 1)", got=str(seen[0].shape))
    ts = [float(s.flat[0]) for s in seen]
    if not (abs(ts[0]) < 1e-6 and all(b > a for a, b in zip(ts, ts[1:]))):
        raise KataFail("t should march 0, 1/N, 2/N, ... over the integration",
                       got=str([round(x, 3) for x in ts]))

    # sampling without explicit noise must be stochastic
    s1 = sample_actions(None, lin_field, obs, jax.random.PRNGKey(0), flow_steps=8, act_dim=ACT_DIM)
    s2 = sample_actions(None, lin_field, obs, jax.random.PRNGKey(1), flow_steps=8, act_dim=ACT_DIM)
    if np.allclose(np.asarray(s1), np.asarray(s2)):
        raise KataFail("different rngs gave identical samples",
                       hint="x_0 should be drawn from the rng when noises is None")


# --------------------------------------------------------------------------
# L08 -- train for real
# --------------------------------------------------------------------------

def check_08():
    train = _get("train", "train")
    result = train(steps=40_000, batch_size=256, seed=0)
    _stub_guard(result, "train")
    if not isinstance(result, dict) or "state" not in result:
        raise KataFail("train() must return a dict containing at least 'state'",
                       got=str(type(result)))
    val = result.get("val_loss", result.get("loss"))
    if val is None:
        raise KataFail("train() should report a final 'val_loss'")
    if not np.isfinite(float(val)):
        raise KataFail("final loss is NaN", got=val)
    if float(val) > 0.20:
        raise KataFail("trained loss is too high to have learned the demos",
                       expected="val_loss < 0.20", got=f"{float(val):.3f}",
                       hint="normalise the observations; 3k steps at lr 3e-4 is plenty")


# --------------------------------------------------------------------------
# L09 -- close the loop
# --------------------------------------------------------------------------

def check_09():
    evaluate_policy = _get("rollout", "evaluate_policy")
    res = evaluate_policy(n_episodes=20, seed=1234)
    _stub_guard(res, "evaluate_policy")
    sr = res["success_rate"] if isinstance(res, dict) else float(res)
    print(f"closed-loop success rate: {sr:.0%}")
    if sr < 0.4:
        raise KataFail("the policy rarely lifts the cube",
                       expected=">= 40% success over 20 episodes", got=f"{sr:.0%}",
                       hint="normalise obs at inference with the TRAINING statistics; "
                            "and check you are not feeding the net un-normalised actions")


# --------------------------------------------------------------------------
# L10 -- how many integration steps do you actually need?
# --------------------------------------------------------------------------

def check_10():
    sweep = _get("rollout", "flow_steps_sweep")
    res = sweep(steps_list=(1, 2, 4, 10), n_episodes=20, seed=1234)
    _stub_guard(res, "flow_steps_sweep")
    if not isinstance(res, dict):
        raise KataFail("flow_steps_sweep must return {flow_steps: success_rate}",
                       got=type(res).__name__)
    missing = [k for k in (1, 2, 4, 10) if k not in res]
    if missing:
        raise KataFail(f"missing entries for flow_steps {missing}")
    print("  ".join(f"{k}:{res[k]:.0%}" for k in sorted(res)))

    if res[10] < 0.6:
        raise KataFail("the 10-step policy should be your good one",
                       expected=">= 60% at flow_steps=10", got=f"{res[10]:.0%}",
                       hint="this is the same policy that passed L09 -- if it got worse, "
                            "your sweep is re-sampling or re-normalising differently")
    if res[1] >= res[10]:
        raise KataFail("a single Euler step did just as well as ten",
                       expected="a clear degradation at flow_steps=1",
                       got=f"1 step: {res[1]:.0%}, 10 steps: {res[10]:.0%}",
                       hint="if one step is as good as ten, your sampler probably is not "
                            "integrating -- check that dt = 1/flow_steps")


# --------------------------------------------------------------------------
# L11 -- why flow matching (vs plain regression)
# --------------------------------------------------------------------------

def check_11():
    fit = _get("diagnostics", "fit_bimodal_toy")
    out = fit()
    _stub_guard(out, "fit_bimodal_toy")
    for k in ("flow_samples", "mse_samples"):
        if k not in out:
            raise KataFail(f"fit_bimodal_toy must return '{k}'")
    flow = np.asarray(out["flow_samples"]).ravel()
    mse = np.asarray(out["mse_samples"]).ravel()

    # The toy target is bimodal at -1 and +1. Regression must collapse to ~0;
    # the flow model must keep both modes.
    if np.abs(mse).mean() > 0.4:
        raise KataFail("the MSE model did not collapse to the mean as expected",
                       expected="|mean prediction| ~ 0", got=f"{np.abs(mse).mean():.3f}",
                       hint="the point of this level is to SEE mode averaging -- "
                            "train a plain deterministic regressor on the same data")
    frac_near_modes = np.mean(np.abs(np.abs(flow) - 1.0) < 0.35)
    if frac_near_modes < 0.6:
        raise KataFail("the flow model did not recover the two modes",
                       expected=">= 60% of samples near -1 or +1",
                       got=f"{frac_near_modes:.0%}",
                       hint="train longer, or sample with more flow_steps")
    print(f"flow kept both modes ({frac_near_modes:.0%} of samples), "
          f"MSE collapsed to {mse.mean():+.3f}")


# --------------------------------------------------------------------------
# L12 -- boss
# --------------------------------------------------------------------------

def check_12():
    ev = _get("rollout", "evaluate_policy")
    res = ev(n_episodes=50, seed=99999)
    sr = res["success_rate"] if isinstance(res, dict) else float(res)
    print(f"final success rate over 50 held-out episodes: {sr:.0%}")
    if sr < 0.85:
        raise KataFail("not yet at expert level",
                       expected=">= 85% over 50 held-out episodes", got=f"{sr:.0%}",
                       hint="more demos, more training steps, obs+action normalisation, "
                            "a wider net, or a longer chunk horizon")


LEVELS = [
    Level(1, "load and batch the demos", f"{SRC}/data.py",
          """
You have 54,000 (observation, action) pairs from 600 successful grasps.
Write `load_demos(path)` and `sample_batch(dataset, rng, batch_size)`.

The only subtle part: a batch must be a *coherent* set of transitions.
Draw one index vector and use it to index every array.
          """,
          check_01,
          hints=["np.load on an .npz gives you a lazy archive -- materialise the arrays you need.",
                 "jax.random.randint(rng, (batch_size,), 0, n) gives you the index vector.",
                 "batch = {k: v[idx] for k, v in dataset.items()} -- one idx, every key."],
          ogpo_refs=["ogpo/utils/datasets.py  -- see the Dataset class and its sample()"]),

    Level(2, "normalise the observations", f"{SRC}/data.py",
          """
Cube positions live around 0.03 m; joint angles swing over radians. Feed that
to a network raw and the small-scale dims are invisible.

Write a `Normalizer` with `fit(x)`, `normalize(x)`, `denormalize(z)`.
Watch out for the obs dims that never change -- they have zero variance.
          """,
          check_02,
          hints=["Store mean and std as arrays of shape (obs_dim,).",
                 "std = x.std(0), then std = np.maximum(std, 1e-6) before dividing.",
                 "A flax.struct.dataclass or a plain NamedTuple both work fine here."],
          ogpo_refs=["ogpo/utils/datasets.py  -- grep for 'normalize'"]),

    Level(3, "the conditional velocity network", f"{SRC}/nets.py",
          """
Flow matching needs v_theta(obs, x_t, t) -> velocity in action space.

Build a flax MLP that takes all three and returns shape (B, action_dim).
The checks will verify the output genuinely depends on each of the three
inputs -- a network that ignores t cannot represent a flow at all.
          """,
          check_03,
          hints=["Simplest thing that works: concatenate [obs, x_t, t] and run an MLP.",
                 "nn.Dense(hidden) + nn.gelu, twice, then nn.Dense(action_dim).",
                 "A learned time embedding (nn.Dense on t) before the concat trains better "
                 "than raw t, and is what OGPO does."],
          ogpo_refs=["ogpo/networks/actors.py:199  -- the __call__ that takes (obs, actions, times)",
                     "ogpo/networks/actors.py:313  -- the time embedding variant"]),

    Level(4, "flow matching targets", f"{SRC}/flow.py",
          """
The heart of it. Given a batch of expert actions x_1:

    x_0 ~ N(0, I)          the noise you start from
    t   ~ U(0, 1)          a random point along the path, PER SAMPLE
    x_t = (1-t) x_0 + t x_1
    v   = x_1 - x_0        the straight-line velocity

Return (x_t, v_target, t). This is the level where broadcasting bites.
          """,
          check_04,
          hints=["t must have shape (batch, 1) so it broadcasts against (batch, action_dim).",
                 "Split your rng: one key for x_0, one for t. Reusing one key correlates them.",
                 "x_1 is just `actions`. There is no network involved in this function at all."],
          ogpo_refs=["ogpo/agents/modules/bc_helper.py:9  -- get_flow_targets, near-identical"]),

    Level(5, "the BC loss", f"{SRC}/flow.py",
          """
    loss = E_{t, x_0} || v_theta(obs, x_t, t) - (x_1 - x_0) ||^2

Write `flow_bc_loss(params, apply_fn, batch, rng)` returning a scalar.

One check here is worth understanding: a model that returns v_target exactly
must score ~0. If your sign or argument order is flipped, it will not.
          """,
          check_05,
          hints=["Call get_flow_targets, then apply_fn(params, obs, x_t, t), then mean-square.",
                 "jnp.mean((pred - v_target) ** 2) -- mean over batch AND action dims.",
                 "Return either a scalar or (scalar, info_dict); the checker accepts both."],
          ogpo_refs=["ogpo/agents/modules/bc_helper.py:119  -- compute_flow_bc_loss"]),

    Level(6, "one gradient step", f"{SRC}/train.py",
          """
Write `create_train_state(rng, obs_dim, act_dim, lr)` and
`update_step(state, batch, rng) -> (new_state, info)`.

The check overfits a single fixed batch 400 times. If the loss does not fall,
something is not connected: the optimiser, the returned state, or the grad.
          """,
          check_06,
          hints=["flax.training.train_state.TrainState.create(apply_fn=, params=, tx=optax.adam(lr)).",
                 "jax.value_and_grad(loss_fn)(state.params), then state.apply_gradients(grads=g).",
                 "Decorate update_step with @jax.jit once it works -- it is ~50x faster."],
          ogpo_refs=["ogpo/agents/modules/flax_utils.py  -- TrainState and apply_loss_fn",
                     "ogpo/agents/ogpo.py  -- grep '_update_offline' for the real thing"]),

    Level(7, "sample by integrating the ODE", f"{SRC}/flow.py",
          """
Training learned a velocity field. To ACT you integrate it:

    x <- x_0 ~ N(0, I)
    repeat flow_steps times:  x <- x + v_theta(obs, x, t) * dt

with dt = 1/flow_steps and t marching 0, 1/N, 2/N, ...

Signature: sample_actions(params, apply_fn, obs, rng, flow_steps, act_dim, noises=None)
          """,
          check_07,
          hints=["t at iteration i is jnp.full((batch, 1), i / flow_steps).",
                 "Clip the final action to [-1, 1] -- the env expects normalised actions.",
                 "`noises` lets a caller supply x_0; when None, draw it from rng."],
          ogpo_refs=["ogpo/agents/modules/pg_helper.py:59  -- sample_flow_actions_ode"]),

    Level(8, "train on the real demos", f"{SRC}/train.py",
          """
Put it together: `train(steps, batch_size, seed)` loads the demos, holds out a
validation split, trains, and returns {'state': ..., 'val_loss': ...}.

Save the trained params to disk -- the next level needs them.
          """,
          check_08,
          hints=["Hold out whole EPISODES, not random transitions -- neighbouring steps "
                 "inside one episode are near-duplicates and would leak.",
                 "Normalise observations with statistics from the TRAIN split only.",
                 "Decay the learning rate (optax.cosine_decay_schedule). This matters more "
                 "than it sounds: at a constant lr the converged policy keeps a "
                 "seed-dependent bias of ~0.2 rad on the finger joints, and closed-loop "
                 "success swings between 0% and 90% at identical validation loss.",
                 "Save params AND the normaliser statistics -- level 9 needs both."],
          ogpo_refs=["ogpo/runners/bc_runner.py  -- the whole offline BC loop"]),

    Level(9, "close the loop on the hand", f"{SRC}/rollout.py",
          """
The payoff. Load your params, run LeapLiftEnv, and act with your policy at
every step. Report a success rate over 20 episodes.

    from wuji_hands.leap_lift import LeapLiftEnv

Also save a video -- watching the failures is how you debug this.
          """,
          check_09,
          hints=["Normalise the observation with the SAME Normalizer you trained with. "
                 "This is the single most common reason a BC policy that trained fine "
                 "does nothing at all in the env.",
                 "env.step wants a (20,) array in [-1, 1]; your net returns (1, 20).",
                 "imageio.mimsave(path, frames, fps=20) with env.render() gives you the video."],
          ogpo_refs=["ogpo/utils/evaluation.py  -- evaluate(), including the video plumbing"]),

    Level(10, "how much ODE do you need?", f"{SRC}/rollout.py",
          """
Sampling an action means integrating an ODE, and every step costs a forward
pass. So how many do you actually need?

Write `flow_steps_sweep(steps_list, n_episodes, seed)` returning
{flow_steps: success_rate}, using the policy you already trained.

Predict the shape of the curve before you run it. Most people are surprised.
          """,
          check_10,
          hints=["Nothing to retrain -- this is the same params, sampled differently.",
                 "flow_steps=1 means a single Euler step from noise: x_1 = x_0 + v(x_0, 0). "
                 "That is a linear map of Gaussian noise, so it can only ever produce a "
                 "Gaussian. Watch it fail.",
                 "More is not monotonically better either. Ask yourself where the extra "
                 "error at very large step counts comes from."],
          ogpo_refs=["ogpo/agents/modules/pg_helper.py:59  -- flow_steps is a parameter there",
                     "ogpo/agents/ogpo.py:2122  -- sample_actions_one_step, the distilled "
                     "one-step policy that exists precisely because 1 naive step fails"]),

    Level(11, "prove flow beats regression", f"{SRC}/diagnostics.py",
          """
Why go to all this trouble instead of a plain MSE regressor?

Build a toy dataset where, for the SAME observation, the expert sometimes acts
-1 and sometimes +1. Fit both a deterministic regressor and your flow model.
Return samples from each as {'flow_samples': ..., 'mse_samples': ...}.

The regressor will confidently output 0 -- an action the expert never took.
          """,
          check_11,
          hints=["obs can be a single constant; the whole point is one obs, two actions.",
                 "The MSE model provably converges to E[a|o] = 0. That is the lesson.",
                 "Your flow code needs no changes -- just action_dim=1."],
          ogpo_refs=["the OGPO paper's motivation section; no code needed for this one"]),

    Level(12, "boss: 85% on held-out seeds", f"{SRC}/rollout.py",
          """
Everything you have, evaluated on 50 unseen cube placements.

The scripted expert gets 98%. Your clone has to get to 85%.
If you are stuck in the 60s, the answer is almost never 'more layers'.
          """,
          check_12,
          hints=["Collect more demos: python -m wuji_hands.collect --episodes 2000.",
                 "Train longer. 40k steps takes about 90 seconds on this box.",
                 "Check WHICH episodes fail -- render them with video_path=. Failures "
                 "cluster at the edges of the cube-position distribution, where the "
                 "demonstrations are sparsest.",
                 "That last point is the covariate-shift argument for going beyond BC, "
                 "which is exactly where OGPO's online RL picks up."]),
]
