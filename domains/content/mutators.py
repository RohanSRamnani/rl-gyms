"""State mutators — the curriculum hook. A TrendShockMutator or
ColdStartMutator slots into a MutatorSequence here later."""
from typing import Any, Dict, Optional

import numpy as np
from rlgym.api import StateMutator

from domains.content.state import (
    AFFINITY_INIT_MAX, AFFINITY_INIT_MIN, INIT_FOLLOWERS_MAX, INIT_FOLLOWERS_MIN,
    N_FORMATS, N_HOOKS, N_TOPICS, SIGNAL_NOISE_STD, ContentState,
)


class CycleStartMutator(StateMutator[ContentState]):
    """Fresh 30-day cycle: new account size, new hidden tables, clean slate."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply(self, state: ContentState, shared_info: Dict[str, Any]) -> None:
        state.followers = float(np.exp(self.rng.uniform(
            np.log(INIT_FOLLOWERS_MIN), np.log(INIT_FOLLOWERS_MAX))))
        state.meta = self.rng.uniform(AFFINITY_INIT_MIN, AFFINITY_INIT_MAX, N_FORMATS)
        state.topic_heat = self.rng.uniform(AFFINITY_INIT_MIN, AFFINITY_INIT_MAX, N_TOPICS)
        state.hook_fit = self.rng.uniform(AFFINITY_INIT_MIN, AFFINITY_INIT_MAX,
                                          (N_FORMATS, N_HOOKS))
        # Day-0 signals: a noisy reading of the fresh hidden state, clipped like
        # the engine's, so the agent never starts blind.
        state.trend_signals = np.clip(
            state.meta + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_FORMATS),
            0.0, None)
        state.heat_signals = np.clip(
            state.topic_heat + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_TOPICS),
            0.0, None)
        state.fatigue = np.zeros(N_FORMATS)
        state.authority = np.zeros(N_TOPICS)
        state.flag_strikes = 0
        state.restricted = False
        state.had_outlier = False
        state.history = []
        state.ticks = 0
        state.last_action = None
        state.repeat_streak = 0
        state.views_today = 0.0
        state.follows_today = 0.0
        state.outlier_today = False
        state.posted_today = False
