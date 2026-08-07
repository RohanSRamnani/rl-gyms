"""Observation builder — turns state into what each agent actually sees."""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ObsBuilder

from domains.minirl.state import (
    BALL_MAX_SPEED, CAR_MAX_SPEED, POS_NORM, MiniRLState, wrap_angle,
)

OBS_SIZE = 16

# Ball-relative velocity spans both speed budgets at once: a ball travelling at
# BALL_MAX_SPEED away from a car travelling at CAR_MAX_SPEED gives 8300 uu/s.
# Normalizing by CAR_MAX_SPEED alone would push this component past 3.6.
REL_VEL_NORM = CAR_MAX_SPEED + BALL_MAX_SPEED


class DefaultObs(ObsBuilder[int, np.ndarray, MiniRLState, int]):
    """A 16-dimensional normalized observation.

    Layout:
        0:2   car position          8:10  ball position
        2:4   car velocity         10:12  ball velocity
        4:6   car heading (cos,sin) 12:14 ball position relative to car
        6:7   car boost            14:16 ball velocity relative to car, then distance

    Every component is normalized so the whole vector stays within [-2, 2];
    relative position is the widest term, reaching exactly 2.0 corner-to-corner.

    Yaw is encoded as (cos, sin) rather than a raw angle: a raw angle has a
    discontinuity at +/-pi that quietly wrecks learning.
    """

    def get_obs_space(self, agent: int) -> int:
        return OBS_SIZE

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[int], state: MiniRLState,
                  shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        return {agent: self._build_one(agent, state) for agent in agents}

    def _build_one(self, agent: int, state: MiniRLState) -> np.ndarray:
        car = state.cars[agent]
        # Orange sees a point-mirrored world, so both teams face the same problem.
        mirror = -1.0 if car.is_orange else 1.0

        car_pos = car.pos * mirror
        car_vel = car.vel * mirror
        ball_pos = state.ball_pos * mirror
        ball_vel = state.ball_vel * mirror
        yaw = wrap_angle(car.yaw + np.pi) if car.is_orange else car.yaw

        rel_pos = ball_pos - car_pos
        rel_vel = ball_vel - car_vel
        distance = float(np.linalg.norm(rel_pos))

        obs = np.concatenate([
            car_pos / POS_NORM,
            car_vel / CAR_MAX_SPEED,
            [np.cos(yaw), np.sin(yaw)],
            [car.boost / 100.0],
            ball_pos / POS_NORM,
            ball_vel / BALL_MAX_SPEED,
            rel_pos / POS_NORM,
            rel_vel / REL_VEL_NORM,
            [distance / 10000.0],
        ])
        return obs.astype(np.float32)
