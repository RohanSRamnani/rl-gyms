"""Per-episode metric tracking. Domain-agnostic."""
from collections import deque
from typing import Dict


class EpisodeTracker:
    """Accumulates per-step signals into per-episode statistics."""

    def __init__(self, window: int = 100) -> None:
        self.returns = deque(maxlen=window)
        self.touches = deque(maxlen=window)
        self.goals = deque(maxlen=window)
        self._current_return = 0.0
        self._current_touched = False

    def record(self, reward: float, terminated: bool, truncated: bool,
               touched: bool, scored: bool) -> None:
        self._current_return += reward
        self._current_touched = self._current_touched or touched
        if terminated or truncated:
            self.returns.append(self._current_return)
            self.touches.append(1.0 if self._current_touched else 0.0)
            self.goals.append(1.0 if scored else 0.0)
            self._current_return = 0.0
            self._current_touched = False

    def snapshot(self) -> Dict[str, float]:
        if not self.returns:
            return {"mean_episode_reward": 0.0, "touch_rate": 0.0,
                    "goal_rate": 0.0, "episodes": 0}
        return {
            "mean_episode_reward": sum(self.returns) / len(self.returns),
            "touch_rate": sum(self.touches) / len(self.touches),
            "goal_rate": sum(self.goals) / len(self.goals),
            "episodes": len(self.returns),
        }
