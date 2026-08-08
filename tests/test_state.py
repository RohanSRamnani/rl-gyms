# tests/test_state.py
import numpy as np
from domains.minirl.state import (
    Car, MiniRLState, wrap_angle, heading,
    SIDE_WALL_X, BACK_WALL_Y, GOAL_HALF_WIDTH, CAR_MAX_SPEED, TICK_RATE, DT,
)


def test_constants_are_consistent():
    assert GOAL_HALF_WIDTH < SIDE_WALL_X
    assert DT == 1.0 / TICK_RATE


def test_wrap_angle_folds_into_pi_range():
    assert np.isclose(wrap_angle(3 * np.pi), np.pi) or np.isclose(wrap_angle(3 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(0.5), 0.5)
    assert -np.pi <= wrap_angle(100.0) <= np.pi


def test_heading_is_unit_vector_along_yaw():
    np.testing.assert_allclose(heading(0.0), [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(heading(np.pi / 2), [0.0, 1.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(heading(1.234)), 1.0)


def test_state_constructs_and_defaults():
    car = Car(pos=np.zeros(2), vel=np.zeros(2), yaw=0.0, boost=100.0, is_orange=False)
    state = MiniRLState(ball_pos=np.zeros(2), ball_vel=np.zeros(2), cars={0: car})
    assert state.ticks == 0
    assert state.last_touch_tick is None
    assert state.scoring_team is None
    assert state.cars[0].boost == 100.0
