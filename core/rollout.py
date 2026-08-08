"""Vectorized rollout collection. Domain-agnostic: takes an env factory."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.metrics import EpisodeTracker
from core.ppo import ActorCritic


class VecEnv:
    """Runs N single-agent RLGym environments in-process with auto-reset.

    On an episode boundary `step` returns the observation of the FINAL state
    before resetting, in `final_obs`. PPO needs that observation to bootstrap
    truncated episodes correctly.
    """

    def __init__(self, env_fn: Callable[[], Any], n_envs: int, agent_id: int = 0) -> None:
        self.envs = [env_fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        self.agent_id = agent_id
        self.last_obs: Optional[np.ndarray] = None   # carried across rollouts
        # Episode-end facts captured BEFORE each auto-reset. The reset clears
        # scoring_team and last_touch_tick, so reading them afterwards always
        # reports None and silently pins goal_rate to 0.
        self._episode_scored: List[Optional[int]] = [None] * n_envs
        self._episode_touched: List[bool] = [False] * n_envs

    def reset(self) -> np.ndarray:
        return np.stack([env.reset()[self.agent_id] for env in self.envs])

    def step(self, actions: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Optional[np.ndarray]]
    ]:
        observations, rewards = [], []
        terminated_flags, truncated_flags = [], []
        final_observations: List[Optional[np.ndarray]] = []

        for index, env in enumerate(self.envs):
            obs, reward, terminated, truncated = env.step(
                {self.agent_id: int(actions[index])}
            )
            done_terminated = bool(terminated[self.agent_id])
            done_truncated = bool(truncated[self.agent_id])

            rewards.append(float(reward[self.agent_id]))
            terminated_flags.append(done_terminated)
            truncated_flags.append(done_truncated)

            # Snapshot before the reset destroys the evidence.
            self._episode_touched[index] = env.state.last_touch_tick is not None
            self._episode_scored[index] = env.state.scoring_team

            if done_terminated or done_truncated:
                final_observations.append(obs[self.agent_id])
                observations.append(env.reset()[self.agent_id])
            else:
                final_observations.append(None)
                observations.append(obs[self.agent_id])

        return (np.stack(observations), np.array(rewards, dtype=np.float32),
                np.array(terminated_flags), np.array(truncated_flags),
                final_observations)

    def scoring_teams(self) -> List[Optional[int]]:
        """Scoring team as of the last `step`, captured before any auto-reset."""
        return list(self._episode_scored)

    def touched_flags(self) -> List[bool]:
        """Whether the ball had been touched as of the last `step`."""
        return list(self._episode_touched)


def collect_rollout(vec_env: VecEnv, model: ActorCritic, horizon: int,
                    tracker: EpisodeTracker) -> Dict[str, np.ndarray]:
    """Collect `horizon` steps from every env, with correct bootstrap values."""
    n = vec_env.n_envs
    obs_size = model.policy[0].in_features

    obs_buffer = np.zeros((horizon, n, obs_size), dtype=np.float32)
    actions_buffer = np.zeros((horizon, n), dtype=np.int64)
    log_probs_buffer = np.zeros((horizon, n), dtype=np.float32)
    values_buffer = np.zeros((horizon, n), dtype=np.float32)
    rewards_buffer = np.zeros((horizon, n), dtype=np.float32)
    bootstraps_buffer = np.zeros((horizon, n), dtype=np.float32)
    dones_buffer = np.zeros((horizon, n), dtype=np.float32)

    if vec_env.last_obs is None:
        vec_env.last_obs = vec_env.reset()
    observations = vec_env.last_obs

    for t in range(horizon):
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        actions, log_probs, values = model.act(obs_tensor)

        obs_buffer[t] = observations
        actions_buffer[t] = actions.numpy()
        log_probs_buffer[t] = log_probs.numpy()
        values_buffer[t] = values.numpy()

        next_obs, rewards, terminated, truncated, final_obs = vec_env.step(
            actions_buffer[t]
        )
        # Both are snapshots taken inside step(), before the auto-reset.
        touched = vec_env.touched_flags()
        scored = [team is not None for team in vec_env.scoring_teams()]

        rewards_buffer[t] = rewards
        dones_buffer[t] = (terminated | truncated).astype(np.float32)

        # The bootstrap value: 0 on TERMINATION, V(final_obs) on TRUNCATION,
        # V(next_obs) otherwise. This is the distinction that keeps value
        # learning honest.
        bootstrap = np.zeros(n, dtype=np.float32)
        ongoing = ~(terminated | truncated)
        if ongoing.any():
            bootstrap[ongoing] = model.predict_values(
                torch.as_tensor(next_obs[ongoing], dtype=torch.float32)
            ).numpy()
        for i in range(n):
            if truncated[i] and not terminated[i]:
                bootstrap[i] = float(model.predict_values(
                    torch.as_tensor(final_obs[i], dtype=torch.float32).unsqueeze(0)
                ).item())
        bootstraps_buffer[t] = bootstrap

        for i in range(n):
            tracker.record(float(rewards[i]), bool(terminated[i]), bool(truncated[i]),
                           touched[i], scored[i])

        observations = next_obs

    vec_env.last_obs = observations
    return {
        "obs": obs_buffer, "actions": actions_buffer, "log_probs": log_probs_buffer,
        "values": values_buffer, "rewards": rewards_buffer,
        "bootstraps": bootstraps_buffer, "dones": dones_buffer,
    }
