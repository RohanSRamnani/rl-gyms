import numpy as np
import pytest
from domains.minirl.state import Car, DT, CAR_MAX_SPEED, MAX_TURN_RATE
from domains.minirl.physics import step_car


def make_car(yaw=0.0, vel=None, boost=100.0):
    return Car(pos=np.zeros(2), vel=np.zeros(2) if vel is None else np.array(vel, dtype=float),
               yaw=yaw, boost=boost, is_orange=False)


def test_full_throttle_accelerates_along_yaw():
    car = make_car(yaw=0.0)
    step_car(car, throttle=1.0, steer=0.0, boost_input=0.0, dt=DT)
    assert car.vel[0] > 0
    assert np.isclose(car.vel[1], 0.0, atol=1e-9)
    assert car.pos[0] > 0


def test_speed_plateaus_at_max_speed():
    car = make_car()
    for _ in range(2000):
        step_car(car, throttle=1.0, steer=0.0, boost_input=0.0, dt=DT)
    assert np.linalg.norm(car.vel) <= CAR_MAX_SPEED + 1e-6
    assert np.linalg.norm(car.vel) > CAR_MAX_SPEED * 0.95


def test_boost_drains_and_stops_at_zero():
    car = make_car(boost=1.0)
    for _ in range(200):
        step_car(car, throttle=1.0, steer=0.0, boost_input=1.0, dt=DT)
    assert car.boost == 0.0


def test_coasting_decelerates_to_exactly_zero():
    car = make_car(vel=[500.0, 0.0])
    for _ in range(300):
        step_car(car, throttle=0.0, steer=0.0, boost_input=0.0, dt=DT)
    np.testing.assert_allclose(car.vel, [0.0, 0.0], atol=1e-6)


def test_turn_rate_degrades_with_speed():
    slow = make_car()
    step_car(slow, throttle=0.0, steer=1.0, boost_input=0.0, dt=DT)
    slow_turn = abs(slow.yaw)

    fast = make_car(vel=[CAR_MAX_SPEED, 0.0])
    step_car(fast, throttle=1.0, steer=1.0, boost_input=0.0, dt=DT)
    fast_turn = abs(fast.yaw)

    assert slow_turn > fast_turn
    assert slow_turn == pytest.approx(MAX_TURN_RATE * DT, rel=1e-6)


def test_yaw_stays_wrapped():
    car = make_car()
    for _ in range(1000):
        step_car(car, throttle=0.0, steer=1.0, boost_input=0.0, dt=DT)
    assert -np.pi <= car.yaw <= np.pi
