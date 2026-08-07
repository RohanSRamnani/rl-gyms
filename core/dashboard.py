"""Live training dashboard. Domain-agnostic: the domain supplies draw_fn."""
from typing import Callable, Dict, List

import matplotlib.pyplot as plt


class Dashboard:
    """Arena on the left, training curves on the right.

    The domain passes in `draw_fn(ax, state)`; core never imports a domain.
    """

    def __init__(self, draw_fn: Callable[[object, object], None],
                 title: str = "MiniRL") -> None:
        self.draw_fn = draw_fn
        self.iterations: List[int] = []
        self.rewards: List[float] = []
        self.touch_rates: List[float] = []
        self.goal_rates: List[float] = []

        plt.ion()
        self.figure, (self.arena_ax, self.curve_ax) = plt.subplots(
            1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.3]}
        )
        self.figure.canvas.manager.set_window_title(title)
        self.rate_ax = self.curve_ax.twinx()

    def update_curve(self, iteration: int, snapshot: Dict[str, float]) -> None:
        self.iterations.append(iteration)
        self.rewards.append(snapshot["mean_episode_reward"])
        self.touch_rates.append(snapshot["touch_rate"])
        self.goal_rates.append(snapshot["goal_rate"])

        self.curve_ax.clear()
        self.rate_ax.clear()
        self.curve_ax.plot(self.iterations, self.rewards, color="tab:blue",
                           label="mean episode reward")
        self.curve_ax.set_xlabel("iteration")
        self.curve_ax.set_ylabel("mean episode reward", color="tab:blue")
        self.curve_ax.grid(alpha=0.3)

        self.rate_ax.plot(self.iterations, self.touch_rates, color="tab:green",
                          linestyle="--", label="touch rate")
        self.rate_ax.plot(self.iterations, self.goal_rates, color="tab:red",
                          linestyle=":", label="goal rate")
        self.rate_ax.set_ylabel("rate")
        self.rate_ax.set_ylim(-0.05, 1.05)

        handles = self.curve_ax.get_legend_handles_labels()[0] + \
            self.rate_ax.get_legend_handles_labels()[0]
        labels = self.curve_ax.get_legend_handles_labels()[1] + \
            self.rate_ax.get_legend_handles_labels()[1]
        self.curve_ax.legend(handles, labels, loc="upper left", fontsize=9)
        self._flush()

    def show_episode(self, states: List[object], frame_pause: float = 0.001) -> None:
        for state in states:
            self.draw_fn(self.arena_ax, state)
            self._flush(frame_pause)

    def _flush(self, pause: float = 0.001) -> None:
        try:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(pause)
        except Exception:
            # Headless backends have no event loop; drawing is enough.
            pass

    def close(self) -> None:
        plt.close(self.figure)
