import numpy as np
from domains.minirl.state import Car, MiniRLState, TICK_RATE
from domains.minirl.dones import (
    AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition,
)


def make_state(ticks=0, last_touch_tick=None, scoring_team=None, n_cars=1):
    cars = {i: Car(pos=np.zeros(2), vel=np.zeros(2), yaw=0.0, boost=100.0,
                   is_orange=bool(i)) for i in range(n_cars)}
    return MiniRLState(ball_pos=np.zeros(2), ball_vel=np.zeros(2), cars=cars,
                       ticks=ticks, last_touch_tick=last_touch_tick,
                       scoring_team=scoring_team)


def test_goal_condition_is_false_before_a_goal():
    condition = GoalCondition()
    state = make_state()
    condition.reset([0], state, {})
    assert condition.is_done([0], state, {}) == {0: False}


def test_goal_condition_is_true_after_a_goal():
    condition = GoalCondition()
    state = make_state(scoring_team=0)
    condition.reset([0], state, {})
    assert condition.is_done([0], state, {}) == {0: True}


def test_conditions_return_an_entry_for_every_agent():
    condition = GoalCondition()
    state = make_state(scoring_team=0, n_cars=2)
    condition.reset([0, 1], state, {})
    assert condition.is_done([0, 1], state, {}) == {0: True, 1: True}


def test_timeout_condition_fires_at_the_limit():
    condition = TimeoutCondition(timeout_seconds=30.0)
    condition.reset([0], make_state(), {})
    assert condition.is_done([0], make_state(ticks=30 * TICK_RATE - 1), {}) == {0: False}
    assert condition.is_done([0], make_state(ticks=30 * TICK_RATE), {}) == {0: True}


def test_no_touch_timeout_measures_from_episode_start_when_never_touched():
    condition = NoTouchTimeoutCondition(timeout_seconds=10.0)
    condition.reset([0], make_state(), {})
    assert condition.is_done([0], make_state(ticks=10 * TICK_RATE - 1), {}) == {0: False}
    assert condition.is_done([0], make_state(ticks=10 * TICK_RATE), {}) == {0: True}


def test_no_touch_timeout_resets_after_a_touch():
    condition = NoTouchTimeoutCondition(timeout_seconds=10.0)
    condition.reset([0], make_state(), {})
    state = make_state(ticks=10 * TICK_RATE, last_touch_tick=9 * TICK_RATE)
    assert condition.is_done([0], state, {}) == {0: False}


def test_any_condition_ors_its_children():
    condition = AnyCondition(
        TimeoutCondition(timeout_seconds=30.0),
        NoTouchTimeoutCondition(timeout_seconds=10.0),
    )
    condition.reset([0], make_state(), {})
    assert condition.is_done([0], make_state(ticks=5), {}) == {0: False}
    assert condition.is_done([0], make_state(ticks=10 * TICK_RATE), {}) == {0: True}
