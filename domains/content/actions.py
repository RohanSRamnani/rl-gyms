"""Action parser. Actions are already engine-shaped ints; this casts and
documents the space: a = format*(N_HOOKS*N_TOPICS) + hook*N_TOPICS + topic,
or SKIP_ACTION."""
from typing import Any, Dict, List

from rlgym.api import ActionParser

from domains.content.state import (
    N_ACTIONS, N_HOOKS, N_TOPICS, SKIP_ACTION, ContentState,
)

__all__ = ["ContentAction", "N_ACTIONS", "SKIP_ACTION", "encode_action"]


def encode_action(format_index: int, hook_index: int, topic_index: int) -> int:
    """(format, hook, topic) -> action int. Inverse of state.decode_action."""
    return format_index * (N_HOOKS * N_TOPICS) + hook_index * N_TOPICS + topic_index


class ContentAction(ActionParser[int, int, int, ContentState, int]):
    def get_action_space(self, agent: int) -> int:
        return N_ACTIONS

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[int, int], state: ContentState,
                      shared_info: Dict[str, Any]) -> Dict[int, int]:
        return {agent: int(action) for agent, action in actions.items()}
