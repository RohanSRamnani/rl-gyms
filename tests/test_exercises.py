"""Stage 3: reimplement the config objects from understanding.

These are SKIPPED until you set MINIRL_EXERCISES=1:

    MINIRL_EXERCISES=1 ./.venv/bin/pytest tests/test_exercises.py -v

The working reference stays in domains/minirl/ — peek if you get stuck.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MINIRL_EXERCISES") != "1",
    reason="set MINIRL_EXERCISES=1 to work through the Stage 3 exercises",
)

from domains.minirl.state import Car, MiniRLState, CAR_MAX_SPEED, TICK_RATE


def make_state(car_pos=(0.0, -1000.0), car_vel=(0.0, 0.0), ball_pos=(0.0, 0.0),
               ball_vel=(0.0, 0.0), ticks=0, last_touch_tick=None, scoring_team=None):
    car = Car(pos=np.array(car_pos, dtype=float), vel=np.array(car_vel, dtype=float),
              yaw=0.0, boost=100.0, is_orange=False)
    return MiniRLState(ball_pos=np.array(ball_pos, dtype=float),
                       ball_vel=np.array(ball_vel, dtype=float), cars={0: car},
                       ticks=ticks, last_touch_tick=last_touch_tick,
                       scoring_team=scoring_team)


def test_your_obs_has_the_right_shape_and_bounds():
    from exercises.obs import DefaultObs
    builder = DefaultObs()
    assert builder.get_obs_space(0) == 16
    obs = builder.build_obs([0], make_state(), {})[0]
    assert obs.shape == (16,)
    assert np.all(np.abs(obs) <= 2.0)


def test_your_speed_toward_ball_reward_has_the_right_sign():
    from exercises.rewards import SpeedTowardBallReward
    reward = SpeedTowardBallReward()
    reward.reset([0], make_state(), {})
    toward = make_state(car_vel=(0.0, CAR_MAX_SPEED))
    away = make_state(car_vel=(0.0, -CAR_MAX_SPEED))
    assert reward.get_rewards([0], toward, {0: False}, {0: False}, {})[0] > 0
    assert reward.get_rewards([0], away, {0: False}, {0: False}, {})[0] == 0.0


def test_your_done_conditions_return_a_dict_per_agent():
    from exercises.dones import GoalCondition, TimeoutCondition
    goal = GoalCondition()
    goal.reset([0], make_state(), {})
    assert goal.is_done([0], make_state(scoring_team=0), {}) == {0: True}
    assert goal.is_done([0], make_state(), {}) == {0: False}

    timeout = TimeoutCondition(timeout_seconds=30.0)
    timeout.reset([0], make_state(), {})
    assert timeout.is_done([0], make_state(ticks=30 * TICK_RATE), {}) == {0: True}
