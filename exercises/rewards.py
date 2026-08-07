"""Stage 3: reimplement the reward functions.

Reference: domains/minirl/rewards.py
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import RewardFunction

from domains.minirl.state import BACK_WALL_Y, BALL_MAX_SPEED, CAR_MAX_SPEED, MiniRLState


class SpeedTowardBallReward(RewardFunction[int, MiniRLState, float]):
    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        # Project the car's velocity onto the direction to the ball, normalize by
        # CAR_MAX_SPEED, and clamp negatives to zero. Watch out for distance == 0.
        raise NotImplementedError


class VelocityBallToGoalReward(RewardFunction[int, MiniRLState, float]):
    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        raise NotImplementedError


class TouchReward(RewardFunction[int, MiniRLState, float]):
    # Hint: this is an EVENT reward. It must fire exactly once per touch, which
    # means remembering the last_touch_tick you already paid out for.
    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        raise NotImplementedError


class GoalReward(RewardFunction[int, MiniRLState, float]):
    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        raise NotImplementedError


class CombinedReward(RewardFunction[int, MiniRLState, float]):
    def __init__(self, *weighted_rewards: Tuple[RewardFunction, float]) -> None:
        self.weighted_rewards = weighted_rewards

    def reset(self, agents: List[int], initial_state: MiniRLState,
              shared_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_rewards(self, agents: List[int], state: MiniRLState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        raise NotImplementedError
