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


def test_contact_ring_uses_the_true_contact_radius():
    # The dashed ring is the animation's authority on when a hit happens, so it
    # must match the threshold physics actually uses — and nothing the renderer
    # does may change the constants physics and rewards depend on.
    import domains.minirl.render as render
    from domains.minirl import state as st
    assert render.CONTACT_RADIUS == st.CAR_RADIUS + st.BALL_RADIUS
    assert st.BALL_RADIUS == 91.25 and st.CAR_RADIUS == 60.0


def _ball_facecolour(ax):
    from domains.minirl.state import BALL_RADIUS
    for patch in ax.patches:
        if getattr(patch, "radius", None) == BALL_RADIUS:
            return patch.get_facecolor()
    raise AssertionError("no ball patch found at BALL_RADIUS")


def test_ball_turns_red_only_on_a_recent_touch():
    figure, ax = plt.subplots()

    quiet = make_state()
    quiet.last_touch_tick = None
    draw_arena(ax, quiet)
    quiet_colour = _ball_facecolour(ax)

    just_hit = make_state()
    just_hit.ticks, just_hit.last_touch_tick = 50, 48
    draw_arena(ax, just_hit)
    hit_colour = _ball_facecolour(ax)

    assert quiet_colour != hit_colour, "the ball must flash on a real touch"
    plt.close(figure)


def test_dashboard_records_curve_history():
    dashboard = Dashboard(draw_fn=draw_arena)
    dashboard.update_curve(1, {"mean_episode_reward": 0.5, "touch_rate": 0.1,
                               "goal_rate": 0.0, "episodes": 4})
    dashboard.update_curve(2, {"mean_episode_reward": 1.5, "touch_rate": 0.4,
                               "goal_rate": 0.1, "episodes": 9})
    assert dashboard.iterations == [1, 2]
    assert dashboard.rewards == [0.5, 1.5]
    assert dashboard.rates["touch"] == [0.1, 0.4]
    assert dashboard.rates["goal"] == [0.0, 0.1]
    dashboard.close()


def test_dashboard_show_episode_accepts_states():
    dashboard = Dashboard(draw_fn=draw_arena)
    dashboard.show_episode([make_state(), make_state()])
    dashboard.close()
