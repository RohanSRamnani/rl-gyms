import numpy as np
from domains.content.state import (
    ContentState, PostRecord, AFFINITY_MAX, CYCLE_TICKS, HISTORY_K,
    MAX_STRIKES, VIEWS_REF,
)
from domains.content.obs import (
    AUTHORITY_SLICE, ContentObs, DAY_COS_IDX, DAY_SIN_IDX, FATIGUE_SLICE,
    FOLLOWERS_IDX, HEAT_SLICE, HISTORY_START, OBS_SIZE, POST_DIMS, STRIKES_IDX,
    TREND_SLICE,
)


def post(fmt=0, hook=0, topic=0, day=0, views=1000.0):
    return PostRecord(format=fmt, hook=hook, topic=topic, day=day,
                      views=views, follows=views * 0.01)


def test_obs_space_and_shape():
    builder = ContentObs()
    assert builder.get_obs_space(0) == OBS_SIZE == 49
    assert HISTORY_START + HISTORY_K * POST_DIMS == OBS_SIZE
    state = ContentState()
    builder.reset([0], state, {})
    obs = builder.build_obs([0], state, {})[0]
    assert obs.shape == (49,) and obs.dtype == np.float32


def test_empty_history_slots_are_zero():
    obs = ContentObs().build_obs([0], ContentState(), {})[0]
    assert np.all(obs[HISTORY_START:] == 0.0)


def test_slot_zero_is_the_most_recent_post():
    state = ContentState()
    state.history = [post(fmt=0, hook=0, topic=0, day=0),
                     post(fmt=2, hook=3, topic=1, day=1, views=VIEWS_REF)]
    obs = ContentObs().build_obs([0], state, {})[0]
    base = HISTORY_START
    assert obs[base + 2] == 1.0                     # format one-hot -> 2
    assert obs[base + 3 + 3] == 1.0                 # hook one-hot -> 3
    assert obs[base + 3 + 4 + 1] == 1.0             # topic one-hot -> 1
    assert np.isclose(obs[base + POST_DIMS - 1], 1.0)   # views at VIEWS_REF
    # Slot 1 is the older post (format 0).
    assert obs[base + POST_DIMS + 0] == 1.0


def test_signals_come_from_signals_not_hidden_state():
    state = ContentState()
    state.meta = np.array([2.0, 2.0, 2.0])
    state.trend_signals = np.array([0.5, 0.5, 0.5])
    obs = ContentObs().build_obs([0], state, {})[0]
    assert np.allclose(obs[TREND_SLICE], 0.5 / AFFINITY_MAX)


def test_strikes_and_authority_visible():
    state = ContentState()
    state.flag_strikes = 2
    state.authority = np.array([0.0, 1.0, 0.5])
    obs = ContentObs().build_obs([0], state, {})[0]
    assert np.isclose(obs[STRIKES_IDX], 2 / MAX_STRIKES)
    assert np.allclose(obs[AUTHORITY_SLICE], [0.0, 1.0, 0.5])


def test_day_encoding_is_continuous_at_the_cycle_boundary():
    builder = ContentObs()
    end_state, start_state = ContentState(), ContentState()
    end_state.ticks = CYCLE_TICKS - 1
    end = builder.build_obs([0], end_state, {})[0]
    start = builder.build_obs([0], start_state, {})[0]
    assert abs(end[DAY_COS_IDX] - start[DAY_COS_IDX]) < 0.25
    assert abs(end[DAY_SIN_IDX] - start[DAY_SIN_IDX]) < 0.25


def test_obs_stays_bounded_on_random_states():
    rng = np.random.default_rng(0)
    builder = ContentObs()
    for _ in range(200):
        state = ContentState()
        state.followers = float(rng.uniform(1.0, 1e7))
        state.ticks = int(rng.integers(0, CYCLE_TICKS))
        state.flag_strikes = int(rng.integers(0, MAX_STRIKES + 1))
        state.trend_signals = rng.uniform(0.0, AFFINITY_MAX, 3)
        state.heat_signals = rng.uniform(0.0, AFFINITY_MAX, 3)
        state.fatigue = rng.uniform(0.0, 10.0, 3)
        state.authority = rng.uniform(0.0, 1.0, 3)
        state.history = [post(fmt=int(rng.integers(3)), hook=int(rng.integers(4)),
                              topic=int(rng.integers(3)), day=d,
                              views=float(rng.uniform(0.0, 1e7)))
                         for d in range(int(rng.integers(0, 8)))]
        obs = builder.build_obs([0], state, {})[0]
        assert np.all(np.isfinite(obs))
        assert np.all(np.abs(obs) <= 1.0 + 1e-6)
