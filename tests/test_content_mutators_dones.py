import numpy as np
from domains.content.state import (
    ContentState, PostRecord, CYCLE_TICKS,
    INIT_FOLLOWERS_MIN, INIT_FOLLOWERS_MAX,
    AFFINITY_INIT_MIN, AFFINITY_INIT_MAX,
)
from domains.content.mutators import CycleStartMutator
from domains.content.dones import RestrictionCondition, CycleEndCondition


def test_cycle_start_randomizes_the_hidden_tables():
    state = ContentState()
    CycleStartMutator(rng=np.random.default_rng(0)).apply(state, {})
    assert INIT_FOLLOWERS_MIN <= state.followers <= INIT_FOLLOWERS_MAX
    for table in (state.meta, state.topic_heat, state.hook_fit.ravel()):
        assert np.all(table >= AFFINITY_INIT_MIN)
        assert np.all(table <= AFFINITY_INIT_MAX)
    # Signals are seeded from the fresh hidden state, not stale defaults.
    assert np.all(state.trend_signals >= 0.0)


def test_cycle_start_clears_a_dirty_state():
    state = ContentState()
    state.ticks, state.flag_strikes, state.restricted = 29, 3, True
    state.had_outlier, state.repeat_streak = True, 5
    state.last_action = (1, 1, 1)
    state.fatigue[:] = 3.0
    state.authority[:] = 1.0
    state.history = [PostRecord(format=0, hook=0, topic=0, day=1,
                                views=9.0, follows=1.0)]
    state.views_today, state.follows_today = 100.0, 5.0
    state.outlier_today, state.posted_today = True, True
    CycleStartMutator(rng=np.random.default_rng(1)).apply(state, {})
    assert state.ticks == 0 and state.flag_strikes == 0
    assert not state.restricted and not state.had_outlier
    assert state.history == [] and state.repeat_streak == 0
    assert state.last_action is None
    assert np.all(state.fatigue == 0.0) and np.all(state.authority == 0.0)
    assert state.views_today == 0.0 and state.follows_today == 0.0
    assert not state.outlier_today and not state.posted_today


def test_cycles_vary_between_resets():
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(10):
        state = ContentState()
        CycleStartMutator(rng=rng).apply(state, {})
        seen.add((round(state.followers, 3), tuple(np.round(state.meta, 6))))
    assert len(seen) > 1


def test_restriction_condition():
    condition = RestrictionCondition()
    clean, flagged = ContentState(), ContentState()
    flagged.restricted = True
    condition.reset([0], clean, {})
    assert condition.is_done([0], clean, {}) == {0: False}
    assert condition.is_done([0], flagged, {}) == {0: True}


def test_cycle_end_condition_fires_at_the_limit():
    condition = CycleEndCondition()
    condition.reset([0], ContentState(), {})
    early, done = ContentState(), ContentState()
    early.ticks, done.ticks = CYCLE_TICKS - 1, CYCLE_TICKS
    assert condition.is_done([0], early, {}) == {0: False}
    assert condition.is_done([0], done, {}) == {0: True}
