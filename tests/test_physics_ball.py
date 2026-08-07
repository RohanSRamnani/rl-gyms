import numpy as np
from domains.minirl.state import (
    MiniRLState, DT, BALL_RADIUS, SIDE_WALL_X, BACK_WALL_Y, BALL_MAX_SPEED,
)
from domains.minirl.physics import step_ball


def make_state(ball_pos=(0.0, 0.0), ball_vel=(0.0, 0.0)):
    return MiniRLState(ball_pos=np.array(ball_pos, dtype=float),
                       ball_vel=np.array(ball_vel, dtype=float), cars={})


def test_ball_at_rest_stays_at_rest():
    state = make_state()
    for _ in range(100):
        step_ball(state, DT)
    np.testing.assert_allclose(state.ball_pos, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(state.ball_vel, [0.0, 0.0], atol=1e-9)


def test_moving_ball_slows_monotonically():
    state = make_state(ball_vel=(1000.0, 0.0))
    speeds = []
    for _ in range(60):
        step_ball(state, DT)
        speeds.append(float(np.linalg.norm(state.ball_vel)))
    assert all(b <= a + 1e-9 for a, b in zip(speeds, speeds[1:]))
    assert speeds[-1] < 1000.0


def test_ball_bounces_off_side_wall_with_reduced_speed():
    state = make_state(ball_pos=(SIDE_WALL_X - BALL_RADIUS - 5.0, 0.0), ball_vel=(1000.0, 0.0))
    for _ in range(10):
        step_ball(state, DT)
    assert state.ball_vel[0] < 0
    assert abs(state.ball_vel[0]) < 1000.0
    assert abs(state.ball_pos[0]) <= SIDE_WALL_X - BALL_RADIUS + 1e-6


def test_ball_into_goal_mouth_scores_for_blue():
    state = make_state(ball_pos=(0.0, BACK_WALL_Y - 100.0), ball_vel=(0.0, 3000.0))
    for _ in range(30):
        step_ball(state, DT)
        if state.scoring_team is not None:
            break
    assert state.scoring_team == 0


def test_ball_into_negative_goal_scores_for_orange():
    state = make_state(ball_pos=(0.0, -BACK_WALL_Y + 100.0), ball_vel=(0.0, -3000.0))
    for _ in range(30):
        step_ball(state, DT)
        if state.scoring_team is not None:
            break
    assert state.scoring_team == 1


def test_ball_outside_goal_mouth_bounces_instead_of_scoring():
    state = make_state(ball_pos=(3000.0, BACK_WALL_Y - 100.0), ball_vel=(0.0, 3000.0))
    for _ in range(30):
        step_ball(state, DT)
    assert state.scoring_team is None
    assert state.ball_vel[1] < 0


def test_ball_speed_is_clamped():
    state = make_state(ball_vel=(BALL_MAX_SPEED * 5, 0.0))
    step_ball(state, DT)
    assert np.linalg.norm(state.ball_vel) <= BALL_MAX_SPEED + 1e-6
