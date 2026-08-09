"""Assembles the six configuration objects into the ContentSim environment."""
from typing import Optional

import numpy as np
from rlgym.api import RLGym

from domains.content.actions import ContentAction, N_ACTIONS, SKIP_ACTION
from domains.content.dones import CycleEndCondition, RestrictionCondition
from domains.content.engine import ContentEngine
from domains.content.mutators import CycleStartMutator
from domains.content.obs import OBS_SIZE, ContentObs
from domains.content.rewards import (
    CombinedReward, FollowGrowthReward, OutlierReward, RestrictionReward,
    ViewsReward,
)
from domains.content.state import ContentState

__all__ = ["build_env", "episode_events", "OBS_SIZE", "N_ACTIONS", "EVENT_NAMES"]

EVENT_NAMES = ("outlier", "restricted")


def episode_events(state: ContentState) -> dict:
    """Episode-level events for core.rollout.VecEnv(event_fn=...)."""
    return {
        "outlier": state.had_outlier,
        "restricted": state.restricted,
    }


def build_env(rng: Optional[np.random.Generator] = None) -> RLGym:
    # The single rng is shared by mutator and engine deliberately — one seed,
    # one cycle. Reproducibility comes from construction order, which is fixed.
    rng = rng if rng is not None else np.random.default_rng()
    return RLGym(
        state_mutator=CycleStartMutator(rng=rng),
        obs_builder=ContentObs(),
        action_parser=ContentAction(),
        reward_fn=CombinedReward(
            (ViewsReward(), 0.01),
            (FollowGrowthReward(), 0.2),
            (OutlierReward(), 0.5),
            (RestrictionReward(), 15.0),
        ),
        transition_engine=ContentEngine(rng=rng),
        termination_cond=RestrictionCondition(),
        truncation_cond=CycleEndCondition(),
    )
