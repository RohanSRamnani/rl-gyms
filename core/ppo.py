"""PPO. Domain-agnostic: sizes are arguments, so this file never imports a domain."""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5


def _mlp(in_size: int, out_size: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_size, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_size),
    )


class ActorCritic(nn.Module):
    """Separate policy and value networks over a discrete action space."""

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.policy = _mlp(obs_size, n_actions, hidden)
        self.value = _mlp(obs_size, 1, hidden)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = torch.distributions.Categorical(logits=self.policy(obs))
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), self.value(obs).squeeze(-1)

    @torch.no_grad()
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs).squeeze(-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = torch.distributions.Categorical(logits=self.policy(obs))
        return (distribution.log_prob(actions), distribution.entropy(),
                self.value(obs).squeeze(-1))


def compute_gae(rewards: np.ndarray, values: np.ndarray, bootstraps: np.ndarray,
                dones: np.ndarray, gamma: float, lam: float
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over arrays shaped (T, N).

    `bootstraps[t]` is the value of whatever state actually followed step t:
    0.0 when the episode TERMINATED, V(final_obs) when it was TRUNCATED, and
    V(obs[t+1]) otherwise. Passing it explicitly is what keeps truncation
    correct — reading values[t+1] would use the next episode's first state.
    """
    horizon = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=rewards.dtype)

    for t in reversed(range(horizon)):
        delta = rewards[t] + gamma * bootstraps[t] - values[t]
        running = delta + gamma * lam * (1.0 - dones[t]) * running
        advantages[t] = running

    return advantages, advantages + values


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer,
               batch: Dict[str, torch.Tensor], config: PPOConfig) -> Dict[str, float]:
    """One PPO update over a flat batch. Returns diagnostics."""
    n = batch["obs"].shape[0]
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(config.epochs):
        for start in range(0, n, config.minibatch_size):
            index = slice(start, start + config.minibatch_size)
            log_probs, entropy, values = model.evaluate(batch["obs"][index],
                                                        batch["actions"][index])

            ratio = torch.exp(log_probs - batch["old_log_probs"][index])
            unclipped = ratio * advantages[index]
            clipped = torch.clamp(ratio, 1.0 - config.clip, 1.0 + config.clip) * advantages[index]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = ((values - batch["returns"][index]) ** 2).mean()
            entropy_mean = entropy.mean()
            loss = (policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_mean)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss)
            stats["value_loss"] += float(value_loss)
            stats["entropy"] += float(entropy_mean)
            updates += 1

    return {key: value / max(updates, 1) for key, value in stats.items()}
