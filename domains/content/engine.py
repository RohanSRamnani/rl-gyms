"""The ContentSim transition engine — the only module that advances the world."""
from typing import Any, Dict, List, Optional

import numpy as np
from rlgym.api import TransitionEngine

from domains.content.state import (
    AFFINITY_MAX, AFFINITY_MIN, AUTHORITY_BOOST, AUTHORITY_DECAY, AUTHORITY_GAIN,
    AUTHORITY_MAX, AUTHORITY_SKIP_DECAY, DRIFT_STD, F0, FATIGUE_DECAY,
    FOLLOW_RATE, JUMP_PROB, MAX_STRIKES, MIN_PRIOR_POSTS, N_FORMATS, N_TOPICS,
    OUTLIER_MULTIPLIER, REACH_EXPONENT, REPEAT_FLAG_STREAK, SIGNAL_NOISE_STD,
    STRIKE_PROB, V0, VIEW_NOISE_SIGMA, ContentState, PostRecord, decode_action,
)


class ContentEngine(TransitionEngine[int, ContentState, int]):
    """One `step` = one simulated day. Engine action is a plain int in [0, 37)."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self._state: ContentState = self.create_base_state()
        self._config: Dict[str, Any] = {}

    @property
    def agents(self) -> List[int]:
        return [0]

    @property
    def max_num_agents(self) -> int:
        return 1

    @property
    def state(self) -> ContentState:
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        self._config = value

    def create_base_state(self) -> ContentState:
        return ContentState()

    def set_state(self, desired_state: ContentState, shared_info: Dict[str, Any]) -> ContentState:
        self._state = desired_state
        return self._state

    def step(self, actions: Dict[int, int], shared_info: Dict[str, Any]) -> ContentState:
        state = self._state
        state.views_today = 0.0
        state.follows_today = 0.0
        state.outlier_today = False
        state.posted_today = False

        self._apply_action(int(actions[0]))
        self._roll_flag()
        self._drift()
        self._decay()
        self._refresh_signals()
        state.ticks += 1
        return state

    def close(self) -> None:
        pass

    # -- internals ------------------------------------------------------------

    def _apply_action(self, action: int) -> None:
        state = self._state
        decoded = decode_action(action)
        if decoded is None:
            state.last_action = None
            state.repeat_streak = 0
            return
        if decoded == state.last_action:
            state.repeat_streak += 1
        else:
            state.last_action = decoded
            state.repeat_streak = 1

        format_index, hook_index, topic_index = decoded
        quality = (state.meta[format_index]
                   * state.topic_heat[topic_index]
                   * state.hook_fit[format_index, hook_index]
                   * (1.0 + AUTHORITY_BOOST * state.authority[topic_index])
                   / (1.0 + state.fatigue[format_index]))
        reach = V0 * (state.followers / F0) ** REACH_EXPONENT
        noise = float(np.exp(self.rng.normal(0.0, VIEW_NOISE_SIGMA)))
        views = reach * quality * noise
        follows = FOLLOW_RATE * views * min(quality, 2.0)

        prior_views = [post.views for post in state.history]
        if (len(prior_views) >= MIN_PRIOR_POSTS
                and views >= OUTLIER_MULTIPLIER * float(np.median(prior_views))):
            state.outlier_today = True
            state.had_outlier = True

        state.history.append(PostRecord(
            format=format_index, hook=hook_index, topic=topic_index,
            day=state.ticks, views=views, follows=follows,
        ))
        state.followers += follows
        state.views_today = views
        state.follows_today = follows
        state.posted_today = True
        state.fatigue[format_index] += 1.0
        state.authority[topic_index] = min(
            AUTHORITY_MAX, state.authority[topic_index] + AUTHORITY_GAIN)

    def _roll_flag(self) -> None:
        state = self._state
        if state.restricted:
            return
        if (state.repeat_streak >= REPEAT_FLAG_STREAK
                and self.rng.random() < STRIKE_PROB):
            state.flag_strikes += 1
            if state.flag_strikes >= MAX_STRIKES:
                state.restricted = True

    def _drift(self) -> None:
        state = self._state
        state.meta = np.clip(
            state.meta + self.rng.normal(0.0, DRIFT_STD, N_FORMATS),
            AFFINITY_MIN, AFFINITY_MAX)
        state.topic_heat = np.clip(
            state.topic_heat + self.rng.normal(0.0, DRIFT_STD, N_TOPICS),
            AFFINITY_MIN, AFFINITY_MAX)
        if self.rng.random() < JUMP_PROB:
            jumped_format = int(self.rng.integers(N_FORMATS))
            state.meta[jumped_format] = float(
                self.rng.uniform(AFFINITY_MIN, AFFINITY_MAX))

    def _decay(self) -> None:
        state = self._state
        state.fatigue *= FATIGUE_DECAY
        decay = AUTHORITY_DECAY if state.posted_today else AUTHORITY_SKIP_DECAY
        state.authority *= decay

    def _refresh_signals(self) -> None:
        state = self._state
        state.trend_signals = np.clip(
            state.meta + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_FORMATS),
            0.0, AFFINITY_MAX)
        state.heat_signals = np.clip(
            state.topic_heat + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_TOPICS),
            0.0, AFFINITY_MAX)
