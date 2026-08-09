import numpy as np
from evaluate import (
    evaluate, make_consistent_agent, make_trend_chaser_agent, random_agent,
)
from domains.content.actions import encode_action
from domains.content.obs import HEAT_SLICE, OBS_SIZE, TREND_SLICE
from domains.content.state import N_ACTIONS, SKIP_ACTION, decode_action


def test_evaluate_returns_the_report_shape():
    rng = np.random.default_rng(0)
    report = evaluate(lambda obs: int(rng.integers(N_ACTIONS)), n_scenarios=5, seed=0)
    assert report["n_scenarios"] == 5
    for key in ("restriction_rate", "outlier_rate", "median_views",
                "mean_final_followers", "mean_reward", "worst"):
        assert key in report
    assert 0.0 <= report["restriction_rate"] <= 1.0
    assert 0.0 <= report["outlier_rate"] <= 1.0
    assert len(report["worst"]) <= 5


def test_evaluate_is_reproducible():
    a = evaluate(lambda obs: SKIP_ACTION, n_scenarios=4, seed=3)
    b = evaluate(lambda obs: SKIP_ACTION, n_scenarios=4, seed=3)
    assert a == b


def test_consistent_agent_rotates_hooks_on_a_fixed_format_and_topic():
    agent = make_consistent_agent(format_index=1, topic_index=2)
    seen_hooks = set()
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for _ in range(8):
        decoded = decode_action(agent(obs))
        assert decoded is not None
        f, h, t = decoded
        assert f == 1 and t == 2
        seen_hooks.add(h)
    assert len(seen_hooks) == 4          # all hooks cycled -> no identical streak


def test_consistent_agent_never_gets_restricted():
    report = evaluate(make_consistent_agent(), n_scenarios=10, seed=0)
    assert report["restriction_rate"] == 0.0


def test_trend_chaser_follows_the_signals():
    rng = np.random.default_rng(0)
    agent = make_trend_chaser_agent(rng)
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[TREND_SLICE] = [0.1, 0.9, 0.2]
    obs[HEAT_SLICE] = [0.2, 0.1, 0.8]
    decoded = decode_action(agent(obs))
    assert decoded is not None
    f, h, t = decoded
    assert f == 1 and t == 2


def test_encode_used_by_baselines_matches_decode():
    assert decode_action(encode_action(2, 3, 1)) == (2, 3, 1)
