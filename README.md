# bc_with_hands

Learn to write a flow-matching behaviour-cloning policy by writing one, for a
16-DoF LEAP hand that has to pick a cube off a table.

Twelve levels. Each one is a small piece of real code plus a check that is picky
enough to catch the mistake it is designed to catch. The first failing level is
the only one you have to think about; everything after it shows as locked.

```bash
make check              # run the ladder
make hint LEVEL=4       # escalating hints, three per level
make peek LEVEL=4       # where OGPO implements the same thing
make brief LEVEL=4      # restate the task
make level LEVEL=4      # run just one level
```

## Layout

```
teacher/wuji_hands/   the simulation, the scripted expert, demo collection
teacher/katas/        the level definitions and the check runner
student/wuji_bc/      the worksheets -- the six files you fill in
data/                 demonstrations and the expert video
checkpoints/          trained policies, small enough to keep in git
videos/               rendered rollouts
```

The split is the point. Everything under `teacher/` is scaffolding: read it if
you want, but you are never asked to change it, and it contains no solutions.
Everything under `student/` is yours.

```bash
git clone git@github.com:leo01110111/bc_with_hands.git
cd bc_with_hands
make demos N=1500        # ~5 min, generates data/leap_lift_demos.npz
make check
```

The runner imports `wuji_bc.data`, `wuji_bc.flow` and so on out of `student/`.
Point it somewhere else with `make check STUDENT=/path/to/another/checkout` if
you want to keep several attempts side by side.

No vendored dependencies and no venvs — `pyproject.toml` and `uv.lock` are the
blueprint, so `uv sync` rebuilds the environment. The demonstrations and the
trained checkpoints are small enough to ship in the repo. The two big vision
arrays are not; see below.

## The datasets

```
leap_lift_demos.npz    1500 eps, state only          -> the ladder
leap_lift_pixels.npz  |  300 eps, states + actions   |
leap_lift_frames.npy  |  300 eps, raw images         |- the vision policy
leap_lift_dinov3_s.npy   300 eps, DINOv3 features    |
```

`leap_lift_pixels.npz` holds no pixels despite the name — it is the states and
actions for the pixel run, and row `i` lines up with row `i` of `frames.npy` and
of the features. The three vision files are one dataset in three pieces.

## Large artifacts

Everything in the repo is under 25 MB. The frame and feature arrays are 15 GiB
together, which is past what GitHub will accept, so they live on the Hub:

```bash
hf download leokswang/wuji-hands-leap-lift --repo-type dataset \
    --local-dir data --include '*.npy'
```

| file | shape | dtype | size |
|---|---|---|---|
| `data/leap_lift_frames.npy` | (27000, 2, 224, 224, 3) | uint8 | 7.57 GiB |
| `data/leap_lift_dinov3_s.npy` | (27000, 2, 197, 384) | float16 | 7.61 GiB |

Both are derived, so downloading is a convenience rather than a requirement:

```bash
.venv/bin/python -m wuji_bc.collect_pixels              # -> leap_lift_frames.npy
.venv-vision/bin/python -m wuji_bc.cache_features       # -> leap_lift_dinov3_s.npy
```

The second one runs under `.venv-vision` — torch and DINOv3, kept apart from the
jax venv on purpose.

## The ladder

| # | level | file |
|---|-------|------|
| 1 | load and batch the demos | `data.py` |
| 2 | normalise the observations | `data.py` |
| 3 | the conditional velocity network | `nets.py` |
| 4 | flow matching targets | `flow.py` |
| 5 | the BC loss | `flow.py` |
| 6 | one gradient step | `train.py` |
| 7 | sample by integrating the ODE | `flow.py` |
| 8 | train on the real demos | `train.py` |
| 9 | close the loop on the hand | `rollout.py` |
| 10 | how much ODE do you need? | `rollout.py` |
| 11 | prove flow beats regression | `diagnostics.py` |
| 12 | boss: 85% on held-out seeds | `rollout.py` |

Levels 1-8 are offline and take seconds to check. Level 9 is where the hand
first moves under your policy. Level 12 is the same policy on 50 unseen cube
placements.

## The environment

`teacher/wuji_hands` — a LEAP hand from `mujoco_menagerie` on a 4-DoF actuated
wrist, with a 5.6 cm cube on a table.

```
obs (31,)  hand_qpos(16) wrist_qpos(4) cube_pos(3) cube_quat(4)
           cube_pos - grasp_centre(3) finger_closure(1)
act (20,)  joint position targets in [-1, 1]:  hand(16) wrist_xyz+yaw(4)
```

20 Hz control, 90-step episodes. Success = cube above 12 cm at the end.

```bash
make demos N=2000       # regenerate demonstrations
```

## Things this environment was deliberately built to teach

The env and the scripted expert went through several rewrites, each forced by a
BC failure. The failures are the curriculum, so they are worth stating.

**The expert is reactive, not scheduled.** The first version was a state machine
on a timer (`if t < 0.42 * T: descend`). It made lovely demonstrations and a
policy that scored **0%**. A cloned policy has no clock; it has to infer the
phase from what it sees. The moment the real hand lagged its position command,
the inferred phase ran slow, the fingers closed late, and the whole sequence
desynchronised. Every transition is now a condition on observable state, which
makes the same lag self-correcting.

**The hand does not always start in the same place.** If the wrist always began
centred and then converged onto the cube, then "command = where my wrist already
is" predicts the expert's action almost perfectly — and it is an integrator with
no restoring force, so a policy that learns it drifts steadily off the table.
Randomising the initial wrist pose breaks that shortcut. You can reproduce the
bug by setting `wrist_xy_range = 0.0` in `ResetOptions` and retraining.

**Nothing the expert does depends on hidden state.** The closing ramp is
anchored to the *measured* finger closure rather than an internal counter. With
a hidden counter, two observationally identical states could map to "stay open"
and "start closing", so the recorded actions were bimodal exactly at the moment
that matters. A policy trained on that either chatters between the two or, if
you average its samples, closes halfway onto nothing.

**There is no clock in the observation.** Not because a clock is unrealistic,
but because with one available the network will use it instead of looking at the
cube.

The general lesson, which is worth more than the specific bugs: **BC can only
clone a function of the observation.** Anything your expert uses that the policy
cannot see — a timer, a plan fixed at reset, an internal counter — turns into
irreducible noise in the training data, concentrated at precisely the moments
that decide success.

**Decay your learning rate.** With a constant lr the converged policy keeps a
seed-dependent bias of roughly 0.2 rad on the finger joints -- the difference
between gripping a 5.6 cm cube and squeezing it out. Closed-loop success swung
between 0% and 90% across training seeds *at identical validation loss*. A
cosine decay took four consecutive seeds to 94-100%. Validation loss is a
strikingly bad predictor of whether this policy works.

## Reference numbers

Measured from the solution used to calibrate the thresholds
(1500 demos, 40k steps, cosine-decayed lr, 4 seeds):

| | |
|---|---|
| scripted expert | 100% at every collection noise level |
| val loss after 40k steps | ~0.031 |
| single-step BC, closed loop | 94-100% (seeds 0-3) |
| flow_steps 1 / 2 / 4 / 10 / 32 | 0% / 16% / 96% / 100% / 96% |
| whole ladder, end to end | 75 s |

Action chunking was tried and is **not** in the ladder: on this task it
consistently underperformed single-step BC (H=4 open-loop 33%, receding 10%,
temporal ensemble 37%, vs 90%). That is not a universal result — chunking earns
its keep with image observations and multimodal human demos — but here the state
is fully observed and the expert is reactive, so executing a chunk open-loop
just throws away the feedback the task depends on. Worth knowing before you
reach for it.

## Setup

Needs jax, flax, optax, mujoco and `mujoco_menagerie` on disk. The `Makefile`
points at an existing venv rather than creating one:

```
PY        = /home/leo/OGPO/.venv/bin/python     # change this
STUDENT   = ./student                           # or pass STUDENT=... per call
MUJOCO_GL = egl                                 # headless rendering
```

The menagerie path is a constant at the top of `teacher/wuji_hands/scene.py`.
