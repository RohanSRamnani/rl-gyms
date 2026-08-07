"""Arena drawing. Knows about cars and balls; knows nothing about training."""
import numpy as np
from matplotlib.patches import Circle, Rectangle

from domains.minirl.state import (
    BACK_WALL_Y, BALL_RADIUS, CAR_RADIUS, GOAL_HALF_WIDTH, SIDE_WALL_X, heading,
)

MARGIN = 400.0


def draw_arena(ax, state) -> None:
    """Draw one frame of MiniRL onto `ax`. Clears the axes first."""
    ax.clear()
    ax.set_xlim(-SIDE_WALL_X - MARGIN, SIDE_WALL_X + MARGIN)
    ax.set_ylim(-BACK_WALL_Y - MARGIN, BACK_WALL_Y + MARGIN)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_patch(Rectangle((-SIDE_WALL_X, -BACK_WALL_Y), 2 * SIDE_WALL_X, 2 * BACK_WALL_Y,
                           fill=False, linewidth=2, edgecolor="0.4"))
    ax.axhline(0.0, color="0.85", linewidth=1, zorder=0)

    for goal_y, colour in ((BACK_WALL_Y, "tab:orange"), (-BACK_WALL_Y, "tab:blue")):
        ax.plot([-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], [goal_y, goal_y],
                color=colour, linewidth=5, solid_capstyle="butt")

    ax.add_patch(Circle(tuple(state.ball_pos), BALL_RADIUS,
                        facecolor="0.15", edgecolor="none", zorder=3))

    for car in state.cars.values():
        colour = "tab:orange" if car.is_orange else "tab:blue"
        ax.add_patch(Circle(tuple(car.pos), CAR_RADIUS, facecolor=colour,
                            edgecolor="none", zorder=4))
        nose = car.pos + heading(car.yaw) * CAR_RADIUS * 2.2
        ax.plot([car.pos[0], nose[0]], [car.pos[1], nose[1]],
                color=colour, linewidth=2, zorder=4)
