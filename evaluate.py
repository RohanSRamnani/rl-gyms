"""Run any (obs) -> action agent through seeded ContentSim scenarios.

This is the demo-asset half of the project: the same six objects that train a
policy also stress-test a finished strategy. Every scenario is seeded, so every
failure in the report replays exactly. An LLM-backed agent_fn that reads a
strategy.md is a drop-in — the interface already fits.
"""
import argparse
import itertools
from typing import Callable, Dict, List

import numpy as np
import torch

from core.ppo import ActorCritic
from domains.content.actions import encode_action
from domains.content.env import build_env
from domains.content.obs import HEAT_SLICE, TREND_SLICE
from domains.content.state import CYCLE_TICKS, N_ACTIONS, N_HOOKS

AgentFn = Callable[[np.ndarray], int]


def random_agent(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(N_ACTIONS))


def make_consistent_agent(format_index: int = 0, topic_index: int = 0) -> AgentFn:
    """Convergence without adaptation: one format, one topic, hooks rotating
    daily (identical posts would trip the flag mechanic — rotation is free)."""
    counter = itertools.count()

    def agent(obs: np.ndarray) -> int:
        hook_index = next(counter) % N_HOOKS
        return encode_action(format_index, hook_index, topic_index)
    return agent


def make_trend_chaser_agent(rng: np.random.Generator) -> AgentFn:
    """Adaptation without convergence: argmax the noisy signals every day."""

    def agent(obs: np.ndarray) -> int:
        format_index = int(np.argmax(obs[TREND_SLICE]))
        topic_index = int(np.argmax(obs[HEAT_SLICE]))
        hook_index = int(rng.integers(N_HOOKS))
        return encode_action(format_index, hook_index, topic_index)
    return agent


def load_policy_agent(checkpoint_path: str) -> AgentFn:
    payload = torch.load(checkpoint_path, weights_only=True)
    model = ActorCritic(payload["obs_size"], payload["n_actions"])
    model.load_state_dict(payload["state_dict"])
    model.eval()

    def agent(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits = model.policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())
    return agent


def evaluate(agent_fn: AgentFn, n_scenarios: int = 200, seed: int = 0) -> Dict:
    seeds = np.random.SeedSequence(seed).spawn(n_scenarios)
    results: List[Dict] = []
    all_views: List[float] = []
    for index in range(n_scenarios):
        env = build_env(rng=np.random.default_rng(seeds[index]))
        obs = env.reset()
        total_reward = 0.0
        for _ in range(CYCLE_TICKS):
            action = agent_fn(obs[0])
            obs, rewards, terminated, truncated = env.step({0: action})
            total_reward += rewards[0]
            if terminated[0] or truncated[0]:
                break
        state = env.state
        all_views.extend(post.views for post in state.history)
        results.append({
            "seed_index": index,
            "reward": round(total_reward, 4),
            "restricted": state.restricted,
            "had_outlier": state.had_outlier,
            "final_followers": state.followers,
        })
    report = {
        "n_scenarios": n_scenarios,
        "restriction_rate": sum(r["restricted"] for r in results) / n_scenarios,
        "outlier_rate": sum(r["had_outlier"] for r in results) / n_scenarios,
        "median_views": float(np.median(all_views)) if all_views else 0.0,
        "mean_final_followers": sum(r["final_followers"] for r in results) / n_scenarios,
        "mean_reward": round(sum(r["reward"] for r in results) / n_scenarios, 4),
        "worst": sorted(
            ({"seed_index": r["seed_index"], "reward": r["reward"],
              "restricted": r["restricted"]} for r in results),
            key=lambda r: r["reward"])[:5],
    }
    return report


def format_report(name: str, report: Dict) -> str:
    lines = [
        f"=== {name} over {report['n_scenarios']} scenarios ===",
        f"restriction rate     : {report['restriction_rate']:.2%}",
        f"outlier rate         : {report['outlier_rate']:.2%}",
        f"median views/post    : {report['median_views']:,.0f}",
        f"mean final followers : {report['mean_final_followers']:,.0f}",
        f"mean reward          : {report['mean_reward']:.3f}",
        "worst scenarios      : " + ", ".join(
            f"#{w['seed_index']}({w['reward']:.1f}{'!' if w['restricted'] else ''})"
            for w in report["worst"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an agent in ContentSim.")
    parser.add_argument("--agent",
                        choices=("random", "consistent", "trend_chaser", "policy"),
                        default="trend_chaser")
    parser.add_argument("--checkpoint", default="checkpoints/content.pt")
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.agent == "random":
        rng = np.random.default_rng(args.seed)
        agent_fn: AgentFn = lambda obs: random_agent(obs, rng)
    elif args.agent == "consistent":
        agent_fn = make_consistent_agent()
    elif args.agent == "trend_chaser":
        agent_fn = make_trend_chaser_agent(np.random.default_rng(args.seed))
    else:
        agent_fn = load_policy_agent(args.checkpoint)

    print(format_report(args.agent, evaluate(agent_fn, args.scenarios, args.seed)))
