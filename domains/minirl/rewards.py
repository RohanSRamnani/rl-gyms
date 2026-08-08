"""Reward functions.

The weights used in env.py form a deliberate ladder spanning three orders of
magnitude. Dense shaping terms are training wheels: they must stay small enough
that farming them can never beat actually scoring.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import RewardFunction

from domains.minirl.state import BACK_WALL_Y, BALL_MAX_SPEED, CAR_MAX_SPEED, MiniRLState


class SpeedTowardBallReward(RewardFunction[int, MiniRLState, float]):
    """Rewards closing speed toward the ball, in [0, 1]."""

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            offset = state.ball_pos - car.pos
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6:
                rewards[agent] = 0.0
                continue
            closing = float(np.dot(car.vel, offset / distance))
            rewards[agent] = max(closing / CAR_MAX_SPEED, 0.0)
        return rewards


class VelocityBallToGoalReward(RewardFunction[int, MiniRLState, float]):
    """Rewards ball velocity toward the opponent goal, in [0, 1]."""

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            goal_y = -BACK_WALL_Y if car.is_orange else BACK_WALL_Y
            offset = np.array([0.0, goal_y]) - state.ball_pos
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6:
                rewards[agent] = 0.0
                continue
            toward = float(np.dot(state.ball_vel, offset / distance))
            rewards[agent] = max(toward / BALL_MAX_SPEED, 0.0)
        return rewards


class TouchReward(RewardFunction[int, MiniRLState, float]):
    """Fires exactly once per ball touch."""

    def __init__(self) -> None:
        self._seen_touch_tick = None

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        self._seen_touch_tick = initial_state.last_touch_tick

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        touched = (state.last_touch_tick is not None
                   and state.last_touch_tick != self._seen_touch_tick)
        self._seen_touch_tick = state.last_touch_tick
        return {agent: 1.0 if touched else 0.0 for agent in agents}


class GoalReward(RewardFunction[int, MiniRLState, float]):
    """+1 for the scoring team, -1 for the conceding team, 0 otherwise."""

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        rewards = {}
        for agent in agents:
            if state.scoring_team is None:
                rewards[agent] = 0.0
            else:
                rewards[agent] = 1.0 if state.scoring_team == state.cars[agent].team else -1.0
        return rewards


class CombinedReward(RewardFunction[int, MiniRLState, float]):
    """Weighted sum of component rewards."""

    def __init__(self, *weighted_rewards: Tuple[RewardFunction, float]) -> None:
        self.weighted_rewards = weighted_rewards

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        for reward_fn, _ in self.weighted_rewards:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        totals = {agent: 0.0 for agent in agents}
        for reward_fn, weight in self.weighted_rewards:
            component = reward_fn.get_rewards(agents, state, is_terminated, is_truncated,
                                              shared_info)
            for agent in agents:
                totals[agent] += weight * component[agent]
        return totals
