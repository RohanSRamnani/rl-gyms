"""Wires the MiniRL domain into core and trains it."""
import argparse
import copy
import itertools
from typing import Dict, List

import numpy as np
import torch

from core.dashboard import Dashboard
from core.metrics import EpisodeTracker
from core.ppo import ActorCritic, PPOConfig, compute_gae, ppo_update
from core.rollout import VecEnv, collect_rollout
from domains.minirl.env import N_ACTIONS, OBS_SIZE, build_env, episode_events
from domains.minirl.render import draw_arena


def _flatten(batch: Dict[str, np.ndarray], advantages: np.ndarray,
             returns: np.ndarray) -> Dict[str, torch.Tensor]:
    return {
        "obs": torch.as_tensor(batch["obs"].reshape(-1, OBS_SIZE), dtype=torch.float32),
        "actions": torch.as_tensor(batch["actions"].reshape(-1), dtype=torch.long),
        "old_log_probs": torch.as_tensor(batch["log_probs"].reshape(-1),
                                         dtype=torch.float32),
        "advantages": torch.as_tensor(advantages.reshape(-1), dtype=torch.float32),
        "returns": torch.as_tensor(returns.reshape(-1), dtype=torch.float32),
    }


def _record_episode(model: ActorCritic, max_steps: int = 450) -> List[object]:
    """Play one episode greedily and capture states for playback."""
    env = build_env()
    obs = env.reset()
    states = [copy.deepcopy(env.state)]
    for _ in range(max_steps):
        with torch.no_grad():
            logits = model.policy(torch.as_tensor(obs[0], dtype=torch.float32).unsqueeze(0))
            action = int(torch.argmax(logits, dim=-1).item())
        obs, _, terminated, truncated = env.step({0: action})
        states.append(copy.deepcopy(env.state))
        if terminated[0] or truncated[0]:
            break
    return states


def run_training(iterations: int = 400, n_envs: int = 8, horizon: int = 256,
                 render_every: int = 10, show_dashboard: bool = True,
                 seed: int = 0) -> List[Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = PPOConfig()
    model = ActorCritic(OBS_SIZE, N_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # np.random.seed does NOT reach np.random.default_rng(), so each env's
    # KickoffMutator would otherwise draw fresh OS entropy and spawns would not
    # reproduce across runs with the same seed. Give every env its own stream.
    env_seeds = np.random.SeedSequence(seed).spawn(n_envs)
    env_counter = itertools.count()
    vec_env = VecEnv(lambda: build_env(rng=np.random.default_rng(env_seeds[next(env_counter)])),
                     n_envs=n_envs, event_fn=episode_events)
    tracker = EpisodeTracker(("touch", "goal"))
    dashboard = Dashboard(draw_fn=draw_arena) if show_dashboard else None

    history: List[Dict[str, float]] = []
    for iteration in range(1, iterations + 1):
        batch = collect_rollout(vec_env, model, horizon=horizon, tracker=tracker)
        advantages, returns = compute_gae(
            batch["rewards"], batch["values"], batch["bootstraps"], batch["dones"],
            gamma=config.gamma, lam=config.lam,
        )
        stats = ppo_update(model, optimizer, _flatten(batch, advantages, returns), config)
        snapshot = tracker.snapshot()
        history.append(snapshot)

        print(f"iter {iteration:4d} | reward {snapshot['mean_episode_reward']:8.3f} "
              f"| touch {snapshot['touch_rate']:5.2f} | goal {snapshot['goal_rate']:5.2f} "
              f"| entropy {stats['entropy']:5.3f} | episodes {snapshot['episodes']}")

        if dashboard is not None:
            dashboard.update_curve(iteration, snapshot)
            if render_every and iteration % render_every == 0:
                dashboard.show_episode(_record_episode(model))

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MiniRL agent.")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_training(iterations=args.iterations, n_envs=args.n_envs, horizon=args.horizon,
                 render_every=args.render_every, show_dashboard=not args.no_dashboard,
                 seed=args.seed)
    if not args.no_dashboard:
        # Only block when there is actually a window to keep open; otherwise a
        # redirected/non-interactive run dies on EOFError at the finish line.
        input("Training finished. Press Enter to close the dashboard.")
