"""core/ must carry NO domain vocabulary. These tests use invented event names."""
import numpy as np

from core.metrics import EpisodeTracker


def test_tracker_reports_a_rate_per_event_name():
    tracker = EpisodeTracker(("frobnicated", "quuxed"))
    tracker.record(1.0, terminated=False, truncated=False,
                   events={"frobnicated": True, "quuxed": False})
    tracker.record(2.0, terminated=True, truncated=False,
                   events={"frobnicated": False, "quuxed": False})
    snap = tracker.snapshot()
    assert snap["episodes"] == 1
    assert np.isclose(snap["mean_episode_reward"], 3.0)
    assert np.isclose(snap["frobnicated_rate"], 1.0)   # OR across the episode
    assert np.isclose(snap["quuxed_rate"], 0.0)


def test_tracker_event_flags_reset_between_episodes():
    tracker = EpisodeTracker(("hit",))
    tracker.record(0.0, True, False, {"hit": True})     # episode 1: hit
    tracker.record(0.0, True, False, {"hit": False})    # episode 2: clean
    assert np.isclose(tracker.snapshot()["hit_rate"], 0.5)


def test_tracker_with_no_event_names_is_valid():
    tracker = EpisodeTracker()
    tracker.record(1.0, True, False, {})
    snap = tracker.snapshot()
    assert snap["episodes"] == 1
    assert set(snap) == {"mean_episode_reward", "episodes"}


def test_core_modules_contain_no_minirl_vocabulary():
    import pathlib
    for name in ("metrics.py", "rollout.py", "dashboard.py"):
        text = (pathlib.Path("core") / name).read_text().lower()
        for word in ("touch", "goal", "scoring_team", "ball", "car"):
            assert word not in text, f"core/{name} still says '{word}'"
