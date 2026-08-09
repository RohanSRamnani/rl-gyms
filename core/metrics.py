"""Per-episode metric tracking. Domain-agnostic: event names come from the caller."""
from collections import deque
from typing import Dict, Sequence


class EpisodeTracker:
    """Accumulates per-step signals into per-episode statistics.

    `event_names` are domain-chosen (e.g. ("outlier", "restricted")). Each
    event is OR-accumulated across an episode and reported as `<name>_rate`
    over the last `window` episodes.
    """

    def __init__(self, event_names: Sequence[str] = (), window: int = 100) -> None:
        self.event_names = tuple(event_names)
        self.returns = deque(maxlen=window)
        self.event_history = {name: deque(maxlen=window) for name in self.event_names}
        self._current_return = 0.0
        self._current_events = {name: False for name in self.event_names}

    def record(self, reward: float, terminated: bool, truncated: bool,
               events: Dict[str, bool]) -> None:
        self._current_return += reward
        for name in self.event_names:
            self._current_events[name] = self._current_events[name] or bool(events.get(name, False))
        if terminated or truncated:
            self.returns.append(self._current_return)
            for name in self.event_names:
                self.event_history[name].append(1.0 if self._current_events[name] else 0.0)
            self._current_return = 0.0
            self._current_events = {name: False for name in self.event_names}

    def snapshot(self) -> Dict[str, float]:
        if not self.returns:
            snap = {"mean_episode_reward": 0.0, "episodes": 0}
            snap.update({f"{name}_rate": 0.0 for name in self.event_names})
            return snap
        snap = {
            "mean_episode_reward": sum(self.returns) / len(self.returns),
            "episodes": len(self.returns),
        }
        for name in self.event_names:
            history = self.event_history[name]
            snap[f"{name}_rate"] = sum(history) / len(history)
        return snap
