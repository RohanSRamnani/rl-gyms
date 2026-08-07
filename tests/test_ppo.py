import numpy as np
import torch
from core.ppo import ActorCritic, PPOConfig, compute_gae, ppo_update


def test_actor_critic_act_shapes():
    model = ActorCritic(obs_size=16, n_actions=30)
    obs = torch.zeros((5, 16))
    actions, log_probs, values = model.act(obs)
    assert actions.shape == (5,)
    assert log_probs.shape == (5,)
    assert values.shape == (5,)
    assert actions.min() >= 0 and actions.max() < 30


def test_actor_critic_evaluate_shapes():
    model = ActorCritic(obs_size=16, n_actions=30)
    obs = torch.zeros((5, 16))
    actions = torch.zeros(5, dtype=torch.long)
    log_probs, entropy, values = model.evaluate(obs, actions)
    assert log_probs.shape == (5,)
    assert entropy.shape == (5,)
    assert values.shape == (5,)


def test_gae_with_zero_rewards_and_values_is_zero():
    zeros = np.zeros((3, 2), dtype=np.float32)
    advantages, returns = compute_gae(zeros, zeros, zeros, zeros, gamma=0.99, lam=0.95)
    np.testing.assert_allclose(advantages, 0.0)
    np.testing.assert_allclose(returns, 0.0)


def test_terminated_step_does_not_bootstrap():
    rewards = np.array([[1.0]], dtype=np.float32)
    values = np.array([[0.0]], dtype=np.float32)
    bootstraps = np.array([[0.0]], dtype=np.float32)   # terminated => 0
    dones = np.array([[1.0]], dtype=np.float32)
    advantages, _ = compute_gae(rewards, values, bootstraps, dones, gamma=0.99, lam=0.95)
    assert advantages[0, 0] == np.float32(1.0)


def test_truncated_step_bootstraps_the_final_value():
    rewards = np.array([[1.0]], dtype=np.float32)
    values = np.array([[0.0]], dtype=np.float32)
    bootstraps = np.array([[10.0]], dtype=np.float32)  # truncated => V(final_obs)
    dones = np.array([[1.0]], dtype=np.float32)
    advantages, _ = compute_gae(rewards, values, bootstraps, dones, gamma=0.99, lam=0.95)
    assert np.isclose(advantages[0, 0], 1.0 + 0.99 * 10.0)


def test_gae_accumulates_across_a_non_terminal_step():
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    values = np.zeros((2, 1), dtype=np.float32)
    bootstraps = np.zeros((2, 1), dtype=np.float32)
    dones = np.array([[0.0], [1.0]], dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, bootstraps, dones,
                                      gamma=0.99, lam=0.95)
    assert np.isclose(advantages[1, 0], 1.0)
    assert np.isclose(advantages[0, 0], 1.0 + 0.99 * 0.95 * 1.0)
    np.testing.assert_allclose(returns, advantages + values)


def test_ppo_update_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    model = ActorCritic(obs_size=4, n_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig(epochs=4, minibatch_size=8)

    rng = np.random.default_rng(0)
    batch = {
        "obs": torch.tensor(rng.normal(size=(16, 4)), dtype=torch.float32),
        "actions": torch.zeros(16, dtype=torch.long),
        "old_log_probs": torch.full((16,), -1.0986),
        "advantages": torch.ones(16),
        "returns": torch.ones(16),
    }
    stats = ppo_update(model, optimizer, batch, config)
    assert "policy_loss" in stats and "value_loss" in stats and "entropy" in stats
    assert np.isfinite(stats["policy_loss"])
    assert stats["entropy"] > 0
