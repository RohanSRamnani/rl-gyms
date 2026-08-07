"""Pure physics functions for MiniRL. No RLGym types appear here."""
import numpy as np

from domains.minirl.state import (
    BOOST_ACCEL, BOOST_CONSUMPTION, BRAKE_ACCEL, CAR_MAX_SPEED, CAR_RADIUS,
    COAST_DECEL, MAX_TURN_RATE, SIDE_WALL_X, BACK_WALL_Y, THROTTLE_ACCEL,
    Car, heading, wrap_angle,
    BALL_DRAG, BALL_MAX_SPEED, BALL_RADIUS, GOAL_HALF_WIDTH, WALL_RESTITUTION,
    MiniRLState,
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


def _clamp_speed(vel: np.ndarray, max_speed: float) -> np.ndarray:
    speed = float(np.linalg.norm(vel))
    if speed > max_speed:
        return vel * (max_speed / speed)
    return vel


def step_ball(state: "MiniRLState", dt: float) -> None:
    """Advance the ball by dt seconds, resolving walls and goals."""
    state.ball_vel = state.ball_vel * ((1.0 - BALL_DRAG) ** dt)
    state.ball_vel = _clamp_speed(state.ball_vel, BALL_MAX_SPEED)
    state.ball_pos = state.ball_pos + state.ball_vel * dt

    # Side walls always bounce.
    limit_x = SIDE_WALL_X - BALL_RADIUS
    if abs(state.ball_pos[0]) > limit_x:
        state.ball_pos[0] = np.sign(state.ball_pos[0]) * limit_x
        state.ball_vel[0] *= -WALL_RESTITUTION

    # Back walls bounce unless the ball is inside the goal mouth.
    limit_y = BACK_WALL_Y - BALL_RADIUS
    if abs(state.ball_pos[1]) > limit_y:
        if abs(state.ball_pos[0]) < GOAL_HALF_WIDTH:
            if state.ball_pos[1] >= BACK_WALL_Y - BALL_RADIUS:
                state.scoring_team = 0     # blue attacks +y
            else:
                state.scoring_team = 1     # orange attacks -y
        else:
            state.ball_pos[1] = np.sign(state.ball_pos[1]) * limit_y
            state.ball_vel[1] *= -WALL_RESTITUTION
