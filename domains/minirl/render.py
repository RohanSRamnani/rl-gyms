"""Arena drawing. Knows about cars and balls; knows nothing about training.

Everything here is drawn at TRUE scale, deliberately.

An earlier version magnified the car and ball 3x so they were easier to see in a
big arena. That made the picture lie: contact really happens when the centres are
CONTACT_RADIUS (151.25 uu) apart, but the inflated circles started overlapping at
453 uu, so the car appeared to drive straight through the ball without moving it.

Two devices replace that magnification, and both mean something exact:

  * the dashed **contact ring** around the ball has radius CONTACT_RADIUS, so the
    instant the car's CENTRE crosses it, a collision occurs;
  * the ball and the ring **flash red** for a few ticks after a real touch.

Note that CONTACT_RADIUS is only 1.66x BALL_RADIUS, so any marker magnification
above 1.66 would bury the ring inside the ball and defeat the whole point. True
scale is the only setting that keeps the ring readable. The car is kept findable
by its long heading stroke rather than by an inflated body.
"""
from matplotlib.patches import Circle, Rectangle

from domains.minirl.state import (
    BACK_WALL_Y, BALL_RADIUS, CAR_RADIUS, GOAL_HALF_WIDTH, SIDE_WALL_X, heading,
)

MARGIN = 400.0

# Centre-to-centre distance at which a car and the ball collide. This mirrors the
# threshold in physics.collide_car_ball; the ring drawn at this radius is what
# makes the animation trustworthy.
CONTACT_RADIUS = CAR_RADIUS + BALL_RADIUS

# How many physics ticks the red touch-flash stays lit.
TOUCH_FLASH_TICKS = 8


def _touched_recently(state) -> bool:
    return (state.last_touch_tick is not None
            and (state.ticks - state.last_touch_tick) <= TOUCH_FLASH_TICKS)


def draw_arena(ax, state) -> None:
    """Draw one frame of MiniRL onto `ax`. Clears the axes first."""
    ax.clear()
    ax.set_xlim(-SIDE_WALL_X - MARGIN, SIDE_WALL_X + MARGIN)
    ax.set_ylim(-BACK_WALL_Y - MARGIN, BACK_WALL_Y + MARGIN)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_patch(Rectangle((-SIDE_WALL_X, -BACK_WALL_Y), 2 * SIDE_WALL_X, 2 * BACK_WALL_Y,
                           fill=False, linewidth=2, edgecolor="0.45"))
    ax.axhline(0.0, color="0.9", linewidth=1, zorder=0)

    for goal_y, colour in ((BACK_WALL_Y, "tab:orange"), (-BACK_WALL_Y, "tab:blue")):
        ax.plot([-GOAL_HALF_WIDTH, GOAL_HALF_WIDTH], [goal_y, goal_y],
                color=colour, linewidth=6, solid_capstyle="butt")

    touched = _touched_recently(state)

    # The authority on contact: car centre inside this ring => collision.
    ax.add_patch(Circle(tuple(state.ball_pos), CONTACT_RADIUS, fill=False,
                        edgecolor=("tab:red" if touched else "0.6"),
                        linestyle=(0, (5, 3)), linewidth=1.6, zorder=2))

    ax.add_patch(Circle(tuple(state.ball_pos), BALL_RADIUS,
                        facecolor=("tab:red" if touched else "0.12"),
                        edgecolor="w", linewidth=1.0, zorder=3))

    for car in state.cars.values():
        colour = "tab:orange" if car.is_orange else "tab:blue"
        # The long heading stroke is what makes a true-scale car findable.
        nose = car.pos + heading(car.yaw) * CAR_RADIUS * 7.0
        ax.plot([car.pos[0], nose[0]], [car.pos[1], nose[1]], color=colour,
                linewidth=3.0, zorder=4, solid_capstyle="round")
        ax.add_patch(Circle(tuple(car.pos), CAR_RADIUS, facecolor=colour,
                            edgecolor="w", linewidth=1.0, zorder=5))
