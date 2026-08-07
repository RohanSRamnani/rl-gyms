"""State types and physical constants for MiniRL.

Units are real Rocket League units (1 uu ~= 1 cm) so that every number here
transfers literally to the real rlgym.rocket_league environment.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# --- Arena geometry ---------------------------------------------------------
SIDE_WALL_X = 4096.0
BACK_WALL_Y = 5120.0
GOAL_HALF_WIDTH = 893.0
BALL_RADIUS = 91.25
CAR_RADIUS = 60.0            # the car is a circle in 2D

# --- Speeds -----------------------------------------------------------------
CAR_MAX_SPEED = 2300.0
BALL_MAX_SPEED = 6000.0

# --- Time -------------------------------------------------------------------
TICK_RATE = 60               # physics ticks per second
DT = 1.0 / TICK_RATE
ACTION_REPEAT = 4            # => 15 decisions/sec, matching real RL's effective rate

# --- Car dynamics -----------------------------------------------------------
THROTTLE_ACCEL = 1600.0
BOOST_ACCEL = 991.667
BRAKE_ACCEL = 3500.0
COAST_DECEL = 525.0
MAX_TURN_RATE = 4.7          # rad/s at standstill; degrades with speed
BOOST_CONSUMPTION = 33.3     # boost units per second

# --- Ball dynamics ----------------------------------------------------------
BALL_DRAG = 0.03             # fraction of speed lost per second
WALL_RESTITUTION = 0.6
BALL_HIT_FORCE = 600.0       # so a stationary car still nudges the ball

# --- Observation normalizer -------------------------------------------------
POS_NORM = np.array([SIDE_WALL_X, BACK_WALL_Y])


def wrap_angle(angle: float) -> float:
    """Fold an angle into [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def heading(yaw: float) -> np.ndarray:
    """Unit vector pointing along yaw."""
    return np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)


@dataclass
class Car:
    pos: np.ndarray          # (2,)
    vel: np.ndarray          # (2,)
    yaw: float
    boost: float             # 0..100
    is_orange: bool

    @property
    def team(self) -> int:
        return 1 if self.is_orange else 0


@dataclass
class MiniRLState:
    ball_pos: np.ndarray                       # (2,)
    ball_vel: np.ndarray                       # (2,)
    cars: Dict[int, Car] = field(default_factory=dict)
    ticks: int = 0
    last_touch_tick: Optional[int] = None
    scoring_team: Optional[int] = None
