"""The MiniRL transition engine — the only module that advances physics."""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import TransitionEngine

from domains.minirl.physics import collide_car_ball, step_ball, step_car
from domains.minirl.state import DT, MiniRLState


class MiniRLEngine(TransitionEngine[int, MiniRLState, np.ndarray]):
    """Advances MiniRL physics.

    An engine action for one agent has shape (n_ticks, 3): one
    [throttle, steer, boost] row per physics tick. `step` walks those rows,
    which is how RepeatAction's tiling reaches the simulator.
    """

    def __init__(self) -> None:
        self._state: MiniRLState = self.create_base_state()
        self._config: Dict[str, Any] = {}

    @property
    def agents(self) -> List[int]:
        return list(self._state.cars.keys())

    @property
    def max_num_agents(self) -> int:
        return 2

    @property
    def state(self) -> MiniRLState:
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        self._config = value

    def create_base_state(self) -> MiniRLState:
        return MiniRLState(ball_pos=np.zeros(2), ball_vel=np.zeros(2), cars={})

    def set_state(self, desired_state: MiniRLState, shared_info: Dict[str, Any]) -> MiniRLState:
        self._state = desired_state
        return self._state

    def step(self, actions: Dict[int, np.ndarray], shared_info: Dict[str, Any]) -> MiniRLState:
        n_ticks = len(next(iter(actions.values())))
        for tick_index in range(n_ticks):
            for agent_id, car in self._state.cars.items():
                throttle, steer, boost = actions[agent_id][tick_index]
                step_car(car, float(throttle), float(steer), float(boost), DT)

            step_ball(self._state, DT)
            for agent_id in self._state.cars:
                collide_car_ball(self._state, agent_id, self._state.ticks)

            self._state.ticks += 1
            if self._state.scoring_team is not None:
                break
        return self._state

    def close(self) -> None:
        pass
