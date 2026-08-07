import numpy as np
from domains.minirl.state import Car, MiniRLState, CAR_MAX_SPEED, BALL_MAX_SPEED
from domains.minirl.rewards import (
    CombinedReward, GoalReward, SpeedTowardBallReward, TouchReward,
    VelocityBallToGoalReward,
)


def make_state(car_pos=(0.0, -1000.0), car_vel=(0.0, 0.0), ball_pos=(0.0, 0.0),
               ball_vel=(0.0, 0.0), is_orange=False):
    car = Car(pos=np.array(car_pos, dtype=float), vel=np.array(car_vel, dtype=float),
              yaw=0.0, boost=100.0, is_orange=is_orange)
    return MiniRLState(ball_pos=np.array(ball_pos, dtype=float),
                       ball_vel=np.array(ball_vel, dtype=float), cars={0: car})


NOT_DONE = ({0: False}, {0: False})


def test_speed_toward_ball_is_positive_when_driving_at_the_ball():
    reward = SpeedTowardBallReward()
    state = make_state(car_vel=(0.0, CAR_MAX_SPEED))
    reward.reset([0], state, {})
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] > 0.9


def test_speed_toward_ball_is_zero_when_driving_away():
    reward = SpeedTowardBallReward()
    state = make_state(car_vel=(0.0, -CAR_MAX_SPEED))
    reward.reset([0], state, {})
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] == 0.0


def test_velocity_ball_to_goal_is_positive_toward_the_opponent_goal():
    reward = VelocityBallToGoalReward()
    state = make_state(ball_vel=(0.0, BALL_MAX_SPEED))
    reward.reset([0], state, {})
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] > 0.9


def test_velocity_ball_to_goal_is_zero_toward_own_goal():
    reward = VelocityBallToGoalReward()
    state = make_state(ball_vel=(0.0, -BALL_MAX_SPEED))
    reward.reset([0], state, {})
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] == 0.0


def test_touch_reward_fires_once_per_touch():
    reward = TouchReward()
    state = make_state()
    reward.reset([0], state, {})
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] == 0.0

    state.last_touch_tick = 10
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] == 1.0
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] == 0.0

    state.last_touch_tick = 25
    assert reward.get_rewards([0], state, *NOT_DONE, {})[0] == 1.0


def test_goal_reward_is_positive_for_scorer_and_negative_for_conceder():
    reward = GoalReward()
    scored = make_state()
    scored.scoring_team = 0
    reward.reset([0], scored, {})
    assert reward.get_rewards([0], scored, {0: True}, {0: False}, {})[0] == 1.0

    conceded = make_state()
    conceded.scoring_team = 1
    reward.reset([0], conceded, {})
    assert reward.get_rewards([0], conceded, {0: True}, {0: False}, {})[0] == -1.0


def test_combined_reward_applies_weights():
    combined = CombinedReward((GoalReward(), 10.0), (TouchReward(), 0.5))
    state = make_state()
    state.scoring_team = 0
    state.last_touch_tick = 3
    combined.reset([0], make_state(), {})
    assert combined.get_rewards([0], state, {0: True}, {0: False}, {})[0] == 10.5


def test_combined_reward_returns_one_entry_per_agent():
    combined = CombinedReward((SpeedTowardBallReward(), 0.01))
    state = make_state()
    combined.reset([0], state, {})
    assert set(combined.get_rewards([0], state, *NOT_DONE, {}).keys()) == {0}
