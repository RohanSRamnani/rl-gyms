"""Done conditions.

GoalCondition TERMINATES: the episode genuinely ended, so V(s') is 0.
The timeouts TRUNCATE: we cut the episode off, so V(s') must be bootstrapped.
Conflating the two silently corrupts value learning.
"""
from typing import Any, Dict, List

from rlgym.api import DoneCondition

from domains.minirl.state import TICK_RATE, MiniRLState


class GoalCondition(DoneCondition[int, MiniRLState]):
    """True once a goal has been scored. This is a TERMINATION."""

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        scored = state.scoring_team is not None
        return {agent: scored for agent in agents}


class TimeoutCondition(DoneCondition[int, MiniRLState]):
    """True once the episode has run too long. This is a TRUNCATION."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_ticks = int(timeout_seconds * TICK_RATE)

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        expired = state.ticks >= self.timeout_ticks
        return {agent: expired for agent in agents}


class NoTouchTimeoutCondition(DoneCondition[int, MiniRLState]):
    """True once nobody has touched the ball for too long. This is a TRUNCATION."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_ticks = int(timeout_seconds * TICK_RATE)

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        last_touch = state.last_touch_tick if state.last_touch_tick is not None else 0
        expired = (state.ticks - last_touch) >= self.timeout_ticks
        return {agent: expired for agent in agents}


class AnyCondition(DoneCondition[int, MiniRLState]):
    """True when any child condition is true."""

    def __init__(self, *conditions: DoneCondition) -> None:
        self.conditions = conditions

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        for condition in self.conditions:
            condition.reset(agents, initial_state, shared_info)

    def is_done(self, agents: List[int], state: MiniRLState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        result = {agent: False for agent in agents}
        for condition in self.conditions:
            child = condition.is_done(agents, state, shared_info)
            for agent in agents:
                result[agent] = result[agent] or child[agent]
        return result
