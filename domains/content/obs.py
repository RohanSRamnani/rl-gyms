"""Observation builder: account status + noisy signals + recent-post history.

The hidden tables (meta, topic_heat, hook_fit) are NEVER read here — the agent
sees only trend_signals/heat_signals, the same way the real system only sees
its research corpus, not the algorithm's internals.
"""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ObsBuilder

from domains.content.state import (
    AFFINITY_MAX, CYCLE_TICKS, HISTORY_K, MAX_FATIGUE_NORM, MAX_FOLLOWERS_NORM,
    MAX_STRIKES, N_FORMATS, N_HOOKS, N_TOPICS, VIEWS_REF, ContentState,
)

FOLLOWERS_IDX = 0
DAY_COS_IDX, DAY_SIN_IDX = 1, 2
STRIKES_IDX = 3
TREND_SLICE = slice(4, 4 + N_FORMATS)
HEAT_SLICE = slice(4 + N_FORMATS, 4 + N_FORMATS + N_TOPICS)
FATIGUE_SLICE = slice(10, 10 + N_FORMATS)
AUTHORITY_SLICE = slice(13, 13 + N_TOPICS)
HISTORY_START = 16
POST_DIMS = N_FORMATS + N_HOOKS + N_TOPICS + 1   # one-hots + log-views norm
OBS_SIZE = HISTORY_START + HISTORY_K * POST_DIMS  # 49


class ContentObs(ObsBuilder[int, np.ndarray, ContentState, int]):
    """49 normalized dims. The day-of-cycle is encoded as (cos, sin): a raw tick
    count would present the cycle wrap as a cliff, the same discontinuity as a
    raw yaw angle in MiniRL.
    """

    def get_obs_space(self, agent: int) -> int:
        return OBS_SIZE

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[int], state: ContentState,
                  shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[FOLLOWERS_IDX] = min(
            np.log1p(max(state.followers, 0.0)) / np.log1p(MAX_FOLLOWERS_NORM), 1.0)
        angle = 2.0 * np.pi * (state.ticks % CYCLE_TICKS) / CYCLE_TICKS
        obs[DAY_COS_IDX] = np.cos(angle)
        obs[DAY_SIN_IDX] = np.sin(angle)
        obs[STRIKES_IDX] = min(state.flag_strikes / MAX_STRIKES, 1.0)
        obs[TREND_SLICE] = np.clip(state.trend_signals / AFFINITY_MAX, 0.0, 1.0)
        obs[HEAT_SLICE] = np.clip(state.heat_signals / AFFINITY_MAX, 0.0, 1.0)
        obs[FATIGUE_SLICE] = np.clip(state.fatigue / MAX_FATIGUE_NORM, 0.0, 1.0)
        obs[AUTHORITY_SLICE] = np.clip(state.authority, 0.0, 1.0)

        offset = HISTORY_START
        for post in reversed(state.history[-HISTORY_K:]):   # most recent first
            obs[offset + post.format] = 1.0
            obs[offset + N_FORMATS + post.hook] = 1.0
            obs[offset + N_FORMATS + N_HOOKS + post.topic] = 1.0
            obs[offset + POST_DIMS - 1] = min(
                np.log1p(max(post.views, 0.0)) / np.log1p(VIEWS_REF), 1.0)
            offset += POST_DIMS
        return {agent: obs.copy() for agent in agents}
