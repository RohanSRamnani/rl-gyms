import numpy as np
import torch
from core.metrics import EpisodeTracker
from core.ppo import ActorCritic
from core.rollout import VecEnv, collect_rollout
from domains.minirl.env import build_env, N_ACTIONS, OBS_SIZE


def test_episode_tracker_averages_completed_episodes():
    tracker = EpisodeTracker()
    tracker.record(1.0, terminated=False, truncated=False, touched=False, scored=False)
    tracker.record(2.0, terminated=False, truncated=True, touched=True, scored=False)
    snapshot = tracker.snapshot()
    assert snapshot["episodes"] == 1
    assert np.isclose(snapshot["mean_episode_reward"], 3.0)
    assert np.isclose(snapshot["touch_rate"], 1.0)
    assert np.isclose(snapshot["goal_rate"], 0.0)


def test_episode_tracker_is_safe_with_no_episodes():
    assert EpisodeTracker().snapshot()["episodes"] == 0


def test_vec_env_reset_returns_stacked_observations():
    vec = VecEnv(build_env, n_envs=3)
    obs = vec.reset()
    assert obs.shape == (3, OBS_SIZE)


def test_vec_env_auto_resets_and_reports_final_obs():
    vec = VecEnv(build_env, n_envs=2)
    vec.reset()
    saw_final_obs = False
    for _ in range(600):
        obs, rewards, terminated, truncated, final_obs = vec.step(np.zeros(2, dtype=int))
        assert obs.shape == (2, OBS_SIZE)
        for i in range(2):
            if terminated[i] or truncated[i]:
                assert final_obs[i] is not None
                assert final_obs[i].shape == (OBS_SIZE,)
                saw_final_obs = True
        if saw_final_obs:
            break
    assert saw_final_obs


def test_collect_rollout_returns_correctly_shaped_arrays():
    torch.manual_seed(0)
    vec = VecEnv(build_env, n_envs=2)
    model = ActorCritic(OBS_SIZE, N_ACTIONS, hidden=32)
    tracker = EpisodeTracker()
    batch = collect_rollout(vec, model, horizon=16, tracker=tracker)

    for key in ("obs", "actions", "log_probs", "values", "rewards", "bootstraps", "dones"):
        assert key in batch, key
    assert batch["obs"].shape == (16, 2, OBS_SIZE)
    assert batch["rewards"].shape == (16, 2)
    assert batch["dones"].shape == (16, 2)
    assert np.all(np.isfinite(batch["rewards"]))
    assert set(np.unique(batch["dones"])).issubset({0.0, 1.0})
