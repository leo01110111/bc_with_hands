"""L09, L10 + L12 -- run the policy on the actual hand."""

import numpy as np


def evaluate_policy(n_episodes: int = 20, seed: int = 0, flow_steps: int = 10,
                    video_path: str | None = None) -> dict:
    """Load trained params, roll out in LeapLiftEnv, return {'success_rate': ...}.

        from wuji_hands.leap_lift import LeapLiftEnv, EPISODE_STEPS

    Pass video_path to save a rollout -- watching the failures is the fastest
    debugging tool you have here.
    """
    # TODO(L09)
    return None


def flow_steps_sweep(steps_list=(1, 2, 4, 10), n_episodes: int = 20, seed: int = 1234) -> dict:
    """Success rate as a function of the number of Euler steps at inference.

    Returns {flow_steps: success_rate}. Same trained params throughout -- the
    only thing changing is how finely you integrate the ODE.
    """
    # TODO(L10)
    return None
