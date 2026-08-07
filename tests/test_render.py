import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.dashboard import Dashboard
from domains.minirl.render import draw_arena
from domains.minirl.state import Car, MiniRLState, SIDE_WALL_X, BACK_WALL_Y


def make_state():
    car = Car(pos=np.array([100.0, -2000.0]), vel=np.zeros(2), yaw=0.5,
              boost=100.0, is_orange=False)
    return MiniRLState(ball_pos=np.array([0.0, 0.0]), ball_vel=np.zeros(2), cars={0: car})


def test_draw_arena_populates_the_axes_and_sets_limits():
    figure, ax = plt.subplots()
    draw_arena(ax, make_state())
    assert len(ax.patches) + len(ax.lines) > 0
    left, right = ax.get_xlim()
    assert left <= -SIDE_WALL_X and right >= SIDE_WALL_X
    bottom, top = ax.get_ylim()
    assert bottom <= -BACK_WALL_Y and top >= BACK_WALL_Y
    plt.close(figure)


def test_draw_arena_clears_between_frames():
    figure, ax = plt.subplots()
    draw_arena(ax, make_state())
    first = len(ax.patches)
    draw_arena(ax, make_state())
    assert len(ax.patches) == first
    plt.close(figure)


def test_display_scale_does_not_leak_into_physics():
    # The renderer may inflate markers for legibility, but nothing it does may
    # change the constants physics and rewards depend on.
    import domains.minirl.render as render
    from domains.minirl import state as st
    assert render.DISPLAY_SCALE > 1.0
    assert st.BALL_RADIUS == 91.25 and st.CAR_RADIUS == 60.0


def test_dashboard_records_curve_history():
    dashboard = Dashboard(draw_fn=draw_arena)
    dashboard.update_curve(1, {"mean_episode_reward": 0.5, "touch_rate": 0.1,
                               "goal_rate": 0.0, "episodes": 4})
    dashboard.update_curve(2, {"mean_episode_reward": 1.5, "touch_rate": 0.4,
                               "goal_rate": 0.1, "episodes": 9})
    assert dashboard.iterations == [1, 2]
    assert dashboard.rewards == [0.5, 1.5]
    assert dashboard.touch_rates == [0.1, 0.4]
    dashboard.close()


def test_dashboard_show_episode_accepts_states():
    dashboard = Dashboard(draw_fn=draw_arena)
    dashboard.show_episode([make_state(), make_state()])
    dashboard.close()
