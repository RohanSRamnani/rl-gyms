import numpy as np
from domains.content.env import (
    EVENT_NAMES, N_ACTIONS, OBS_SIZE, build_env, episode_events,
)
from domains.content.state import CYCLE_TICKS, SKIP_ACTION


def test_reset_and_spaces():
    env = build_env(rng=np.random.default_rng(0))
    obs = env.reset()
    assert set(obs.keys()) == {0} and obs[0].shape == (OBS_SIZE,)
    assert env.action_space(0) == N_ACTIONS
    assert env.observation_space(0) == OBS_SIZE


def test_step_returns_four_tuple():
    env = build_env(rng=np.random.default_rng(0))
    env.reset()
    result = env.step({0: 0})
    assert len(result) == 4
    obs, rewards, terminated, truncated = result
    assert isinstance(rewards[0], float)


def test_random_cycle_is_stable_and_finite():
    rng = np.random.default_rng(1)
    env = build_env(rng=rng)
    env.reset()
    steps = 0
    for _ in range(CYCLE_TICKS * 3):
        obs, rewards, terminated, truncated = env.step({0: int(rng.integers(N_ACTIONS))})
        assert np.all(np.isfinite(obs[0])) and np.isfinite(rewards[0])
        steps += 1
        if terminated[0] or truncated[0]:
            break
    assert steps <= CYCLE_TICKS


def test_spamming_one_post_ends_badly_or_times_out():
    env = build_env(rng=np.random.default_rng(2))
    env.reset()
    for _ in range(CYCLE_TICKS):
        obs, rewards, terminated, truncated = env.step({0: 0})
        if terminated[0]:
            assert env.state.restricted
            return
        if truncated[0]:
            return
    raise AssertionError("episode never ended")


def test_skipping_all_cycle_truncates_cleanly():
    env = build_env(rng=np.random.default_rng(3))
    env.reset()
    for _ in range(CYCLE_TICKS):
        obs, rewards, terminated, truncated = env.step({0: SKIP_ACTION})
        assert not terminated[0]                    # skipping can't get you banned
        if truncated[0]:
            assert env.state.ticks == CYCLE_TICKS
            return
    raise AssertionError("episode never truncated")


def test_episode_events_names_and_semantics():
    env = build_env(rng=np.random.default_rng(0))
    env.reset()
    events = episode_events(env.state)
    assert set(events) == set(EVENT_NAMES)
    assert events["outlier"] is False and events["restricted"] is False
    env.state.had_outlier = True
    assert episode_events(env.state)["outlier"] is True
    env.state.restricted = True
    assert episode_events(env.state)["restricted"] is True


def test_seeded_envs_reproduce():
    def run(seed):
        env = build_env(rng=np.random.default_rng(seed))
        env.reset()
        rewards = []
        for step in range(CYCLE_TICKS):
            _, r, term, trunc = env.step({0: step % N_ACTIONS})
            rewards.append(r[0])
            if term[0] or trunc[0]:
                break
        return rewards
    assert run(5) == run(5)
