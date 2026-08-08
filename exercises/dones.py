"""Stage 3: reimplement the done conditions.

Reference: domains/minirl/dones.py

Remember: is_done takes a LIST of agents and returns a DICT keyed by agent.
The official RLGym docs show a different, wrong signature.
"""
from typing import Any, Dict, List

from rlgym.api import DoneCondition

from domains.minirl.state import TICK_RATE, MiniRLState


class GoalCondition(DoneCondition[int, MiniRLState]):
    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        raise NotImplementedError


class TimeoutCondition(DoneCondition[int, MiniRLState]):
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_ticks = int(timeout_seconds * TICK_RATE)

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        raise NotImplementedError


class NoTouchTimeoutCondition(DoneCondition[int, MiniRLState]):
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_ticks = int(timeout_seconds * TICK_RATE)

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        raise NotImplementedError


class AnyCondition(DoneCondition[int, MiniRLState]):
    def __init__(self, *conditions: DoneCondition) -> None:
        self.conditions = conditions

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        raise NotImplementedError
