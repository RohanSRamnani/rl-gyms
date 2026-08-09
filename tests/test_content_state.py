import numpy as np
from domains.content.state import (
    PostRecord, ContentState, decode_action,
    N_FORMATS, N_HOOKS, N_TOPICS, N_ACTIONS, SKIP_ACTION, HISTORY_K, CYCLE_TICKS,
    V0, F0, REACH_EXPONENT, VIEW_NOISE_SIGMA, FOLLOW_RATE,
    INIT_FOLLOWERS_MIN, INIT_FOLLOWERS_MAX,
    AFFINITY_MIN, AFFINITY_MAX, AFFINITY_INIT_MIN, AFFINITY_INIT_MAX,
    DRIFT_STD, JUMP_PROB, SIGNAL_NOISE_STD,
    FATIGUE_DECAY, MAX_FATIGUE_NORM,
    AUTHORITY_GAIN, AUTHORITY_DECAY, AUTHORITY_SKIP_DECAY, AUTHORITY_MAX, AUTHORITY_BOOST,
    REPEAT_FLAG_STREAK, STRIKE_PROB, MAX_STRIKES,
    OUTLIER_MULTIPLIER, MIN_PRIOR_POSTS,
    MAX_FOLLOWERS_NORM, VIEWS_REF, FOLLOWS_REF,
)


def test_constants_are_coherent():
    assert N_ACTIONS == N_FORMATS * N_HOOKS * N_TOPICS + 1 == 37
    assert SKIP_ACTION == N_ACTIONS - 1
    assert CYCLE_TICKS == 30 and HISTORY_K == 3
    assert AFFINITY_MIN < AFFINITY_INIT_MIN < AFFINITY_INIT_MAX < AFFINITY_MAX
    assert 0 < STRIKE_PROB < 1 and 0 < JUMP_PROB < 1
    assert 0 < FATIGUE_DECAY < 1
    assert 0 < AUTHORITY_SKIP_DECAY < AUTHORITY_DECAY < 1
    assert INIT_FOLLOWERS_MIN < INIT_FOLLOWERS_MAX < MAX_FOLLOWERS_NORM
    assert MIN_PRIOR_POSTS >= 1 and OUTLIER_MULTIPLIER > 1.0
    assert REPEAT_FLAG_STREAK >= 2 and MAX_STRIKES >= 1


def test_decode_action_roundtrips_every_post_action():
    seen = set()
    for action in range(N_ACTIONS - 1):
        decoded = decode_action(action)
        assert decoded is not None
        f, h, t = decoded
        assert 0 <= f < N_FORMATS and 0 <= h < N_HOOKS and 0 <= t < N_TOPICS
        seen.add(decoded)
    assert len(seen) == N_FORMATS * N_HOOKS * N_TOPICS


def test_decode_action_skip_is_none():
    assert decode_action(SKIP_ACTION) is None


def test_state_constructs_with_defaults():
    state = ContentState()
    assert state.ticks == 0 and state.flag_strikes == 0
    assert state.restricted is False and state.had_outlier is False
    assert state.history == [] and state.posted_today is False
    assert state.views_today == 0.0 and state.follows_today == 0.0
    assert state.meta.shape == (N_FORMATS,)
    assert state.topic_heat.shape == (N_TOPICS,)
    assert state.hook_fit.shape == (N_FORMATS, N_HOOKS)
    assert state.trend_signals.shape == (N_FORMATS,)
    assert state.heat_signals.shape == (N_TOPICS,)
    assert state.fatigue.shape == (N_FORMATS,)
    assert state.authority.shape == (N_TOPICS,)
    assert state.last_action is None and state.repeat_streak == 0


def test_post_record_fields():
    record = PostRecord(format=1, hook=2, topic=0, day=5, views=1234.0, follows=12.0)
    assert (record.format, record.hook, record.topic) == (1, 2, 0)
    assert record.day == 5 and record.views == 1234.0 and record.follows == 12.0
