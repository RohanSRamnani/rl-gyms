import numpy as np
from domains.minirl.state import (
    Car, MiniRLState, BALL_RADIUS, CAR_RADIUS, BALL_MAX_SPEED,
)
from domains.minirl.physics import collide_car_ball


def make_state(car_pos, ball_pos, car_vel=(0.0, 0.0), ball_vel=(0.0, 0.0), yaw=0.0):
    car = Car(pos=np.array(car_pos, dtype=float), vel=np.array(car_vel, dtype=float),
              yaw=yaw, boost=100.0, is_orange=False)
    return MiniRLState(ball_pos=np.array(ball_pos, dtype=float),
                       ball_vel=np.array(ball_vel, dtype=float), cars={0: car})


def test_no_contact_when_far_apart():
    state = make_state((0.0, 0.0), (1000.0, 0.0))
    assert collide_car_ball(state, 0, tick=5) is False
    assert state.last_touch_tick is None


def test_contact_records_touch_tick():
    state = make_state((0.0, 0.0), (CAR_RADIUS + BALL_RADIUS - 10.0, 0.0))
    assert collide_car_ball(state, 0, tick=7) is True
    assert state.last_touch_tick == 7


def test_overlap_is_resolved():
    state = make_state((0.0, 0.0), (10.0, 0.0))
    collide_car_ball(state, 0, tick=0)
    dist = np.linalg.norm(state.ball_pos - state.cars[0].pos)
    assert dist >= CAR_RADIUS + BALL_RADIUS - 1e-6


def test_stationary_car_still_nudges_ball():
    state = make_state((0.0, 0.0), (CAR_RADIUS + BALL_RADIUS - 5.0, 0.0))
    collide_car_ball(state, 0, tick=0)
    assert state.ball_vel[0] > 0


def test_moving_car_sends_ball_forward_faster_than_stationary_car():
    slow = make_state((0.0, 0.0), (CAR_RADIUS + BALL_RADIUS - 5.0, 0.0))
    collide_car_ball(slow, 0, tick=0)

    fast = make_state((0.0, 0.0), (CAR_RADIUS + BALL_RADIUS - 5.0, 0.0), car_vel=(1500.0, 0.0))
    collide_car_ball(fast, 0, tick=0)

    assert fast.ball_vel[0] > slow.ball_vel[0]


def test_collision_never_exceeds_max_ball_speed():
    state = make_state((0.0, 0.0), (CAR_RADIUS + BALL_RADIUS - 5.0, 0.0),
                       car_vel=(2300.0, 0.0), ball_vel=(-5000.0, 0.0))
    collide_car_ball(state, 0, tick=0)
    assert np.linalg.norm(state.ball_vel) <= BALL_MAX_SPEED + 1e-6
