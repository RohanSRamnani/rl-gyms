"""RestrictionCondition TERMINATES (the account is dead; there is no tomorrow
worth valuing). CycleEndCondition TRUNCATES (the cycle ended, the account did
not — the critic must value the ongoing state: followers carry forward)."""
from typing import Any, Dict, List

from rlgym.api import DoneCondition

from domains.content.state import CYCLE_TICKS, ContentState


class RestrictionCondition(DoneCondition[int, ContentState]):
    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: ContentState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        dead = state.restricted
        return {agent: dead for agent in agents}


class CycleEndCondition(DoneCondition[int, ContentState]):
    def __init__(self, cycle_ticks: int = CYCLE_TICKS) -> None:
        self.cycle_ticks = cycle_ticks

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: ContentState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        over = state.ticks >= self.cycle_ticks
        return {agent: over for agent in agents}
