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


def _fixed_batch(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "obs": torch.tensor(rng.normal(size=(16, 4)), dtype=torch.float32),
        "actions": torch.zeros(16, dtype=torch.long),
        "old_log_probs": torch.full((16,), -1.0986),
        "advantages": torch.tensor(rng.normal(size=16), dtype=torch.float32),
        "returns": torch.tensor(rng.normal(size=16), dtype=torch.float32),
    }


def test_ppo_update_returns_finite_diagnostics():
    torch.manual_seed(0)
    model = ActorCritic(obs_size=4, n_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    stats = ppo_update(model, optimizer, _fixed_batch(),
                       PPOConfig(epochs=4, minibatch_size=8))
    assert set(stats) == {"policy_loss", "value_loss", "entropy"}
    assert all(np.isfinite(v) for v in stats.values())
    assert stats["entropy"] > 0


def test_ppo_update_actually_learns_reducing_value_loss():
    # The critic must fit `returns` better each pass over the same batch.
    # Without this, ppo_update could be a no-op and every other test would
    # still pass — they only check that it runs.
    torch.manual_seed(0)
    model = ActorCritic(obs_size=4, n_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig(epochs=4, minibatch_size=8)
    batch = _fixed_batch()

    losses = [ppo_update(model, optimizer, batch, config)["value_loss"]
              for _ in range(3)]
    assert losses[1] < losses[0], f"value loss did not fall: {losses}"
    assert losses[2] < losses[1], f"value loss did not fall: {losses}"


def test_ppo_update_moves_the_policy_in_the_right_direction():
    # Guards the sign of the clipped surrogate. Every other test in this file
    # passes even with the policy loss sign flipped, because the critic trains
    # through its own MSE term regardless — and a flipped sign trains the agent
    # to get WORSE, which presents only as "training mysteriously fails".
    torch.manual_seed(0)
    model = ActorCritic(obs_size=4, n_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    config = PPOConfig(epochs=4, minibatch_size=8)

    obs = torch.zeros((16, 4))
    batch = {
        "obs": obs,
        "actions": torch.tensor([0, 1] * 8),          # action 0 rewarded,
        "advantages": torch.tensor([1.0, -1.0] * 8),  # action 1 punished
        "old_log_probs": torch.full((16,), -1.0986),
        "returns": torch.zeros(16),
    }

    def log_probs():
        with torch.no_grad():
            return torch.log_softmax(model.policy(obs[:1]), dim=-1)[0]

    before = log_probs()
    for _ in range(5):
        ppo_update(model, optimizer, batch, config)
    after = log_probs()

    assert after[0] > before[0], "rewarded action did not become more likely"
    assert after[1] < before[1], "punished action did not become less likely"


def test_ppo_update_survives_constant_advantages():
    # std == 0 in the normalizer; the 1e-8 guard must stop a NaN escaping.
    torch.manual_seed(0)
    model = ActorCritic(obs_size=4, n_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch = _fixed_batch()
    batch["advantages"] = torch.ones(16)
    stats = ppo_update(model, optimizer, batch, PPOConfig(epochs=2, minibatch_size=8))
    assert all(np.isfinite(v) for v in stats.values())
