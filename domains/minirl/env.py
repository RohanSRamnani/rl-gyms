"""Assembles the six configuration objects into an RLGym environment.

Read alongside the official build_rlgym_v2_env() from the Rocket League docs —
this is deliberately the same shape.
"""
from typing import Optional

import numpy as np
from rlgym.api import RLGym

from domains.minirl.actions import N_ACTIONS, LookupTableAction, RepeatAction
from domains.minirl.dones import (
    AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition,
)
from domains.minirl.engine import MiniRLEngine
from domains.minirl.mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
from domains.minirl.obs import OBS_SIZE, DefaultObs
from domains.minirl.rewards import (
    CombinedReward, GoalReward, SpeedTowardBallReward, TouchReward,
    VelocityBallToGoalReward,
)
from domains.minirl.state import ACTION_REPEAT

__all__ = ["build_env", "OBS_SIZE", "N_ACTIONS"]

NO_TOUCH_TIMEOUT_SECONDS = 10.0
GAME_TIMEOUT_SECONDS = 30.0


def build_env(blue_size: int = 1, orange_size: int = 0,
              rng: Optional[np.random.Generator] = None) -> RLGym:
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size),
            KickoffMutator(rng=rng),
        ),
        obs_builder=DefaultObs(),
        action_parser=RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT),
        reward_fn=CombinedReward(
            (SpeedTowardBallReward(), 0.01),
            (VelocityBallToGoalReward(), 0.1),
            (TouchReward(), 0.5),
            (GoalReward(), 10.0),
        ),
        transition_engine=MiniRLEngine(),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=NO_TOUCH_TIMEOUT_SECONDS),
            TimeoutCondition(timeout_seconds=GAME_TIMEOUT_SECONDS),
        ),
    )
