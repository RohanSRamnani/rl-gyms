import numpy as np
import pytest
from domains.minirl.env import build_env, N_ACTIONS, OBS_SIZE


def test_reset_returns_a_dict_of_observations():
    env = build_env()
    obs = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {0}
    assert obs[0].shape == (OBS_SIZE,)


def test_step_returns_a_four_tuple():
    env = build_env()
    env.reset()
    result = env.step({0: 0})
    assert len(result) == 4, "RLGym.step returns 4 values, not the Gym 5-tuple"
    obs, rewards, terminated, truncated = result
    assert set(obs.keys()) == {0}
    assert isinstance(rewards[0], float)
    assert isinstance(terminated[0], bool)
    assert isinstance(truncated[0], bool)


def test_action_and_observation_spaces():
    env = build_env()
    env.reset()
    assert env.action_space(0) == N_ACTIONS
    assert env.observation_space(0) == OBS_SIZE


def test_random_rollout_is_stable_and_never_produces_nan():
    rng = np.random.default_rng(0)
    env = build_env(rng=rng)
    env.reset()
    for _ in range(1000):
        obs, rewards, terminated, truncated = env.step({0: int(rng.integers(N_ACTIONS))})
        assert np.all(np.isfinite(obs[0]))
        assert np.isfinite(rewards[0])
        if terminated[0] or truncated[0]:
            obs = env.reset()
            assert np.all(np.isfinite(obs[0]))


def test_episodes_always_end_within_the_timeout():
    rng = np.random.default_rng(1)
    env = build_env(rng=rng)
    env.reset()
    steps = 0
    for _ in range(2000):
        _, _, terminated, truncated = env.step({0: int(rng.integers(N_ACTIONS))})
        steps += 1
        if terminated[0] or truncated[0]:
            break
    assert steps < 500, "an episode ran past its truncation limit"


def test_one_v_one_produces_two_agents():
    env = build_env(blue_size=1, orange_size=1)
    obs = env.reset()
    assert set(obs.keys()) == {0, 1}
    obs, rewards, terminated, truncated = env.step({0: 0, 1: 0})
    assert set(rewards.keys()) == {0, 1}
