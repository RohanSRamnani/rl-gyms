"""Action parsers — these define the vocabulary the policy speaks in."""
import itertools
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ActionParser

from domains.minirl.state import ACTION_REPEAT, MiniRLState

THROTTLE_OPTIONS = (-1.0, 0.0, 1.0)
STEER_OPTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)
BOOST_OPTIONS = (0.0, 1.0)
N_ACTIONS = len(THROTTLE_OPTIONS) * len(STEER_OPTIONS) * len(BOOST_OPTIONS)  # 30


class LookupTableAction(ActionParser[int, int, np.ndarray, MiniRLState, int]):
    """Maps a discrete action index onto a [throttle, steer, boost] triple.

    Discretizing keeps the policy a clean categorical distribution, which is
    both easier to learn and easier to reason about than continuous control.
    """

    def __init__(self) -> None:
        self._table = np.array(
            list(itertools.product(THROTTLE_OPTIONS, STEER_OPTIONS, BOOST_OPTIONS)),
            dtype=np.float32,
        )

    def get_action_space(self, agent: int) -> int:
        return N_ACTIONS

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[int, int], state: MiniRLState,
                      shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        return {agent: self._table[int(action)].copy() for agent, action in actions.items()}


class RepeatAction(ActionParser[int, int, np.ndarray, MiniRLState, int]):
    """Holds one decision for `repeats` physics ticks.

    This shortens the effective horizon by `repeats` and is one of the highest
    leverage knobs in the whole configuration.
    """

    def __init__(self, parser: ActionParser, repeats: int = ACTION_REPEAT) -> None:
        self.parser = parser
        self.repeats = repeats

    def get_action_space(self, agent: int) -> int:
        return self.parser.get_action_space(agent)

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        self.parser.reset(agents, initial_state, shared_info)

    def parse_actions(self, actions: Dict[int, int], state: MiniRLState,
                      shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        parsed = self.parser.parse_actions(actions, state, shared_info)
        return {agent: np.tile(control, (self.repeats, 1))
                for agent, control in parsed.items()}
