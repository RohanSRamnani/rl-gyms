import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.dashboard import Dashboard
from domains.content.render import draw_board
from domains.content.state import ContentState, PostRecord


def make_state(n_posts=5, restricted=False):
    state = ContentState()
    state.history = [
        PostRecord(format=i % 3, hook=i % 4, topic=i % 3, day=i,
                   views=float(100 * (i + 1) ** 2), follows=float(i))
        for i in range(n_posts)
    ]
    state.followers = 8000.0
    state.ticks = n_posts
    state.flag_strikes = 1
    state.restricted = restricted
    return state


def test_draw_board_populates_axes_and_clears_between_frames():
    figure, ax = plt.subplots()
    draw_board(ax, make_state())
    first = len(ax.patches) + len(ax.lines) + len(ax.texts)
    assert first > 0
    draw_board(ax, make_state())
    assert len(ax.patches) + len(ax.lines) + len(ax.texts) == first
    plt.close(figure)


def test_draw_board_handles_a_fresh_account():
    figure, ax = plt.subplots()
    draw_board(ax, ContentState())
    plt.close(figure)


def test_dashboard_accepts_draw_board():
    dashboard = Dashboard(draw_fn=draw_board, title="ContentSim")
    dashboard.show_episode([make_state(), make_state(restricted=True)])
    dashboard.update_curve(1, {"mean_episode_reward": 1.0, "episodes": 5,
                               "outlier_rate": 0.3, "restricted_rate": 0.1})
    assert dashboard.rates["outlier"] == [0.3]
    dashboard.close()
