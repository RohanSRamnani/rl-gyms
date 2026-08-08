"""State mutators — these write the initial state, and are where curriculum lives."""
from typing import Any, Dict, Optional

import numpy as np
from rlgym.api import StateMutator

from domains.minirl.state import BACK_WALL_Y, SIDE_WALL_X, Car, MiniRLState


class FixedTeamSizeMutator(StateMutator[MiniRLState]):
    """Populates state.cars with the requested number of cars per team."""

    def __init__(self, blue_size: int = 1, orange_size: int = 0) -> None:
        self.blue_size = blue_size
        self.orange_size = orange_size

    def apply(self, state: MiniRLState, shared_info: Dict[str, Any]) -> None:
        state.cars = {}
        agent_id = 0
        for is_orange in (False, True):
            count = self.orange_size if is_orange else self.blue_size
            for _ in range(count):
                state.cars[agent_id] = Car(pos=np.zeros(2), vel=np.zeros(2), yaw=0.0,
                                           boost=100.0, is_orange=is_orange)
                agent_id += 1


class KickoffMutator(StateMutator[MiniRLState]):
    """Centres the ball and scatters each car randomly within its own half.

    Randomized spawns stop the agent memorizing a single trajectory, and this is
    the hook where a curriculum would later live.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply(self, state: MiniRLState, shared_info: Dict[str, Any]) -> None:
        state.ball_pos = np.zeros(2)
        state.ball_vel = np.zeros(2)
        state.ticks = 0
        state.last_touch_tick = None
        state.scoring_team = None

        for car in state.cars.values():
            side = -1.0 if not car.is_orange else 1.0
            x = self.rng.uniform(-0.75 * SIDE_WALL_X, 0.75 * SIDE_WALL_X)
            y = side * self.rng.uniform(0.2 * BACK_WALL_Y, 0.8 * BACK_WALL_Y)
            car.pos = np.array([x, y], dtype=float)
            car.vel = np.zeros(2)
            car.yaw = float(self.rng.uniform(-np.pi, np.pi))
            car.boost = 100.0


class MutatorSequence(StateMutator[MiniRLState]):
    """Applies mutators in order."""

    def __init__(self, *mutators: StateMutator[MiniRLState]) -> None:
        self.mutators = mutators

    def apply(self, state: MiniRLState, shared_info: Dict[str, Any]) -> None:
        for mutator in self.mutators:
            mutator.apply(state, shared_info)
