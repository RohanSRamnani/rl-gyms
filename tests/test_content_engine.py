import numpy as np
from domains.content.state import (
    ContentState, PostRecord, decode_action,
    N_ACTIONS, SKIP_ACTION, MAX_STRIKES, MIN_PRIOR_POSTS, CYCLE_TICKS,
    AFFINITY_MIN, AFFINITY_MAX,
)
from domains.content.engine import ContentEngine


def seeded_engine(seed=0, **overrides):
    engine = ContentEngine(rng=np.random.default_rng(seed))
    state = engine.create_base_state()
    for key, value in overrides.items():
        setattr(state, key, value)
    engine.set_state(state, {})
    return engine


POST_A = 0                       # (format 0, hook 0, topic 0)
POST_B = 1 * 12 + 2 * 3 + 1      # (format 1, hook 2, topic 1) = action 19


def test_agents_and_max():
    engine = seeded_engine()
    assert engine.agents == [0] and engine.max_num_agents == 1


def test_posting_records_views_and_grows_followers():
    engine = seeded_engine()
    before = engine.state.followers
    engine.step({0: POST_A}, {})
    state = engine.state
    assert state.posted_today and state.views_today > 0.0
    assert len(state.history) == 1
    assert state.history[0].views == state.views_today
    assert state.followers > before
    assert state.fatigue[0] > 0.0 and state.authority[0] > 0.0
    assert state.ticks == 1


def test_skip_posts_nothing_and_decays_authority_harder():
    posted = seeded_engine()
    posted.state.authority[:] = 0.5
    posted.step({0: POST_A}, {})
    skipped = seeded_engine()
    skipped.state.authority[:] = 0.5
    skipped.step({0: SKIP_ACTION}, {})
    assert skipped.state.views_today == 0.0 and skipped.state.history == []
    assert not skipped.state.posted_today
    # topic 2 was posted by neither engine; skip-day decay must bite harder.
    assert skipped.state.authority[2] < posted.state.authority[2]


def test_fatigue_lowers_views_all_else_equal():
    fresh = seeded_engine(seed=3)
    fresh.step({0: POST_A}, {})
    tired = seeded_engine(seed=3)
    tired.state.fatigue[0] = 3.0
    tired.step({0: POST_A}, {})
    assert tired.state.views_today < fresh.state.views_today


def test_authority_raises_views_all_else_equal():
    plain = seeded_engine(seed=4)
    plain.step({0: POST_A}, {})
    authoritative = seeded_engine(seed=4)
    authoritative.state.authority[0] = 1.0
    authoritative.step({0: POST_A}, {})
    assert authoritative.state.views_today > plain.state.views_today


def test_identical_streak_eventually_restricts():
    engine = seeded_engine(seed=0)
    for _ in range(60):
        engine.step({0: POST_A}, {})
        if engine.state.restricted:
            break
    assert engine.state.restricted and engine.state.flag_strikes >= MAX_STRIKES


def test_varied_posting_never_strikes():
    engine = seeded_engine(seed=0)
    for day in range(60):
        engine.step({0: POST_A if day % 2 == 0 else POST_B}, {})
    assert engine.state.flag_strikes == 0 and not engine.state.restricted


def test_outlier_fires_against_trailing_median():
    engine = seeded_engine(seed=1)
    engine.state.history = [
        PostRecord(format=0, hook=0, topic=0, day=d, views=1.0, follows=0.0)
        for d in range(MIN_PRIOR_POSTS)
    ]
    engine.step({0: POST_A}, {})
    # Median of priors is 1.0; any real post clears 2x that by orders of magnitude.
    assert engine.state.outlier_today and engine.state.had_outlier


def test_no_outlier_without_enough_priors():
    engine = seeded_engine(seed=1)
    engine.step({0: POST_A}, {})
    assert not engine.state.outlier_today


def test_hidden_state_drifts_and_stays_clipped():
    engine = seeded_engine(seed=2)
    start_meta = engine.state.meta.copy()
    for _ in range(CYCLE_TICKS):
        engine.step({0: SKIP_ACTION}, {})
    assert not np.allclose(engine.state.meta, start_meta)
    assert np.all(engine.state.meta >= AFFINITY_MIN)
    assert np.all(engine.state.meta <= AFFINITY_MAX)
    assert np.all(engine.state.topic_heat >= AFFINITY_MIN)
    assert np.all(engine.state.topic_heat <= AFFINITY_MAX)


def test_signals_track_hidden_state_noisily():
    engine = seeded_engine(seed=5)
    engine.step({0: SKIP_ACTION}, {})
    state = engine.state
    assert not np.array_equal(state.trend_signals, state.meta)   # noisy...
    assert np.all(np.abs(state.trend_signals - state.meta) < 1.0)  # ...but tethered


def test_same_seed_same_cycle():
    def run(seed):
        engine = seeded_engine(seed=seed)
        for day in range(CYCLE_TICKS):
            engine.step({0: day % N_ACTIONS}, {})
        s = engine.state
        return (round(s.followers, 6), len(s.history), s.flag_strikes,
                tuple(np.round(s.meta, 9)))
    assert run(7) == run(7)
    assert run(7) != run(8)
