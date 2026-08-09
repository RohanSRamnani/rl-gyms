"""Reward functions. Same ladder shape as MiniRL: dense terms are training
wheels and must never outweigh the terminal event.

Built-in hacking demo for Stage-3-style experiments: reward RAW views instead
of this ladder and watch the agent chase every trend signal into fatigue and
flag strikes — Goodhart with a ring light.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import RewardFunction

from domains.content.state import FOLLOWS_REF, VIEWS_REF, ContentState


class ViewsReward(RewardFunction[int, ContentState, float]):
    """Dense-ish: log(1+views_today)/log(1+VIEWS_REF), clipped at 1. Zero on
    skip days. The log + clip keeps the heavy-tailed noise from swamping PPO."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        value = min(np.log1p(max(state.views_today, 0.0)) / np.log1p(VIEWS_REF), 1.0)
        return {agent: float(value) for agent in agents}


class FollowGrowthReward(RewardFunction[int, ContentState, float]):
    """Event: follows gained today / FOLLOWS_REF. The compounding signal."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        return {agent: state.follows_today / FOLLOWS_REF for agent in agents}


class OutlierReward(RewardFunction[int, ContentState, float]):
    """Event: +1 the day a post clears 2x the trailing median — the same
    definition analyze/outliers.py uses in the real system."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        value = 1.0 if state.outlier_today else 0.0
        return {agent: value for agent in agents}


class RestrictionReward(RewardFunction[int, ContentState, float]):
    """Terminal: -1 when the account gets restricted. The episode ends with it."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        value = -1.0 if state.restricted else 0.0
        return {agent: value for agent in agents}


class CombinedReward(RewardFunction[int, ContentState, float]):
    """Weighted sum. Deliberate per-domain copy: domains are self-contained."""

    def __init__(self, *weighted_rewards: Tuple[RewardFunction, float]) -> None:
        self.weighted_rewards = weighted_rewards

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        for reward_fn, _ in self.weighted_rewards:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        totals = {agent: 0.0 for agent in agents}
        for reward_fn, weight in self.weighted_rewards:
            component = reward_fn.get_rewards(agents, state, is_terminated,
                                              is_truncated, shared_info)
            for agent in agents:
                totals[agent] += weight * component[agent]
        return totals
