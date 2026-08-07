"""Pure physics functions for MiniRL. No RLGym types appear here."""
import numpy as np

from domains.minirl.state import (
    BOOST_ACCEL, BOOST_CONSUMPTION, BRAKE_ACCEL, CAR_MAX_SPEED, CAR_RADIUS,
    COAST_DECEL, MAX_TURN_RATE, SIDE_WALL_X, BACK_WALL_Y, THROTTLE_ACCEL,
    Car, heading, wrap_angle,
)


def step_car(car: Car, throttle: float, steer: float, boost_input: float, dt: float) -> None:
    """Advance one car by dt seconds. Mutates `car` in place.

    The car is an arcade car: velocity always points along its heading.
    """
    fwd = heading(car.yaw)
    forward_speed = float(np.dot(car.vel, fwd))

    # Turning degrades with speed — the one handling behaviour worth preserving.
    turn_scale = 1.0 - 0.5 * min(abs(forward_speed) / CAR_MAX_SPEED, 1.0)
    car.yaw = wrap_angle(car.yaw + steer * MAX_TURN_RATE * turn_scale * dt)
    fwd = heading(car.yaw)

    boosting = boost_input > 0.5 and car.boost > 0.0
    if boosting:
        car.boost = max(0.0, car.boost - BOOST_CONSUMPTION * dt)
        accel = THROTTLE_ACCEL + BOOST_ACCEL
    elif throttle == 0.0:
        # Coast toward zero, but never overshoot into reverse.
        decel = COAST_DECEL * dt
        accel = (-forward_speed / dt) if abs(forward_speed) < decel \
            else -np.sign(forward_speed) * COAST_DECEL
    elif forward_speed * throttle < 0.0:
        accel = throttle * BRAKE_ACCEL
    else:
        accel = throttle * THROTTLE_ACCEL

    forward_speed = float(np.clip(forward_speed + accel * dt, -CAR_MAX_SPEED, CAR_MAX_SPEED))
    car.vel = fwd * forward_speed
    car.pos = car.pos + car.vel * dt
    _clamp_car_to_arena(car)


def _clamp_car_to_arena(car: Car) -> None:
    limit_x = SIDE_WALL_X - CAR_RADIUS
    limit_y = BACK_WALL_Y - CAR_RADIUS
    for axis, limit in ((0, limit_x), (1, limit_y)):
        if abs(car.pos[axis]) > limit:
            car.pos[axis] = np.sign(car.pos[axis]) * limit
            car.vel[axis] = 0.0
