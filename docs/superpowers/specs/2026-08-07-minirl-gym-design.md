# MiniRL — a 2D Rocket League for learning the RLGym v2 API

**Date:** 2026-08-07
**Status:** Approved, ready for implementation planning

## Goal

Teach the RLGym v2 environment-design API by building a complete, working, visually
legible environment from scratch — then rebuilding its most important parts from memory.

The learner already understands what an RL gym is conceptually. This project targets the
next layer: how to *design* an environment, and why each design choice changes learning.

## Learning objectives

By the end, the learner can independently:

1. Decompose a control problem into the six RLGym configuration objects.
2. Design a normalized, ego-centric observation space and explain why it trains faster.
3. Build a reward ladder whose dense shaping terms cannot be farmed.
4. Explain and correctly implement the `terminated` vs `truncated` distinction, including
   the value-bootstrap consequence.
5. Read a training curve and distinguish learning, plateau, and reward hacking.
6. Map every object in this project onto its `rlgym.rocket_league` counterpart.

## Non-goals

Explicitly out of scope. Each is cut to keep the project finishable:

- Faithful Rocket League physics. Ours is *plausible*, not accurate. No aerials, no
  flips, no wall driving, no demolitions, no boost pads.
- 3D. The arena is strictly 2D top-down.
- Self-play or an opponent in Stages 1–3.
- GPU training, distributed rollout workers, or hyperparameter search.
- Competitive performance. Success is visible learning, not a good bot.

## Environment: MiniRL

Top-down 2D arena. One car, one ball, two goals. Drive into the ball, knock it into the
opponent goal.

### Architecture

Package layout deliberately mirrors `rlgym.rocket_league` file-for-file, so that reading
the official Rocket League docs afterwards requires no new mapping.

```
minirl/
  state.py      MiniRLState, Car dataclasses; physical constants
  engine.py     MiniRLEngine(TransitionEngine)   — the only module that knows physics
  mutators.py   FixedTeamSizeMutator, KickoffMutator, MutatorSequence
  obs.py        DefaultObs(ObsBuilder)
  actions.py    LookupTableAction, RepeatAction (ActionParser)
  rewards.py    CombinedReward, SpeedTowardBallReward, VelocityBallToGoalReward,
                TouchReward, GoalReward
  dones.py      GoalCondition, TimeoutCondition, NoTouchTimeoutCondition, AnyCondition
  render.py     MatplotlibRenderer(Renderer)
  env.py        build_env() factory
ppo.py          PPO implementation
train.py        training loop + live dashboard
exercises/      Stage 3: stubbed config objects + tests
tests/          pytest suite
```

Class names intentionally match the real `rlgym.rocket_league` names.

### Units and constants

Real Rocket League units (1 uu ~= 1 cm) so values transfer literally.

| Constant | Value |
|---|---|
| `SIDE_WALL_X` | 4096 |
| `BACK_WALL_Y` | 5120 |
| `GOAL_HALF_WIDTH` | 893 |
| `CAR_MAX_SPEED` | 2300 |
| `BALL_MAX_SPEED` | 6000 |
| `BALL_RADIUS` | 91.25 |
| `CAR_RADIUS` | 60 (car approximated as a circle in 2D) |
| `TICK_RATE` | 60 Hz |
| `ACTION_REPEAT` | 4 (=> 15 decisions/sec, matching RL's effective rate) |

Note: real Rocket League runs at 120 Hz with `repeats=8`. We use 60 Hz with `repeats=4`
to halve physics cost while preserving the same 15 Hz decision rate. This difference is
documented in code because the decision rate — not the tick rate — is what shapes the
learning problem.

### State

```python
@dataclass
class Car:
    pos: np.ndarray      # (2,)
    vel: np.ndarray      # (2,)
    yaw: float
    boost: float         # 0..100
    is_orange: bool

@dataclass
class MiniRLState:
    ball_pos: np.ndarray     # (2,)
    ball_vel: np.ndarray     # (2,)
    cars: Dict[AgentID, Car]
    ticks: int
    last_touch_tick: Optional[int]
    scoring_team: Optional[int]   # None until a goal is scored
```

### Physics (`MiniRLEngine`)

Per tick, `dt = 1/60`:

**Car** — engine action is `(throttle, steer, boost)`:
- Forward accel along `yaw`: `throttle * THROTTLE_ACCEL` (1600), plus `BOOST_ACCEL`
  (991.667) when boosting and boost remains.
- Braking (throttle opposing velocity) uses `BRAKE_ACCEL` (3500); zero throttle coasts
  at `COAST_DECEL` (525).
- Yaw rate: `steer * MAX_TURN_RATE * (1 - 0.5 * speed / CAR_MAX_SPEED)` — turning
  degrades with speed, which is the single most important handling behaviour to preserve.
- Speed clamped to `CAR_MAX_SPEED`. Boost drains at 33.3/sec.
- Car clamped inside the arena walls; wall contact zeroes the normal velocity component.

**Ball**:
- Drag: `vel *= (1 - 0.03) ** (dt * 60)`.
- Wall bounce with restitution 0.6, except within the goal mouth.
- Speed clamped to `BALL_MAX_SPEED`.

**Car/ball collision** — circle-circle, triggered when distance < `CAR_RADIUS +
BALL_RADIUS`:
- Positions separated along the contact normal to remove overlap.
- Impulse applied to the ball along the normal, scaled by closing speed, with an added
  constant "hit force" so that a stationary car still nudges the ball.
- Sets `last_touch_tick`.

**Goal**: ball crosses `|y| >= BACK_WALL_Y` while `|x| < GOAL_HALF_WIDTH` sets
`scoring_team`.

Physics is deliberately simple. It must be *stable* and *legible*, not accurate. The spec
treats any behaviour that is stable and looks plausible on screen as correct.

### StateMutator

- `FixedTeamSizeMutator(blue_size=1, orange_size=0)` — populates `state.cars`.
- `KickoffMutator` — ball at centre, stationary; car at a randomized position in its own
  half with randomized yaw and full boost.
- `MutatorSequence` — applies mutators in order.

Randomized spawns (rather than a fixed kickoff) matter here: they prevent the agent from
memorizing one trajectory, and they are the hook where curriculum would later live.

### ObsBuilder

`DefaultObs` — 16 dimensions, all normalized to roughly [-1, 1], mirroring the
coefficient style of the real `DefaultObs`:

| Fields | Dims | Normalizer |
|---|---|---|
| car pos | 2 | `[SIDE_WALL_X, BACK_WALL_Y]` |
| car vel | 2 | `CAR_MAX_SPEED` |
| car yaw as `(cos, sin)` | 2 | — |
| car boost | 1 | 100 |
| ball pos | 2 | `[SIDE_WALL_X, BACK_WALL_Y]` |
| ball vel | 2 | `BALL_MAX_SPEED` |
| ball pos relative to car | 2 | `[SIDE_WALL_X, BACK_WALL_Y]` |
| ball vel relative to car | 2 | `CAR_MAX_SPEED` |
| distance to ball | 1 | 10000 |

Yaw is encoded as `(cos, sin)` rather than a raw angle to avoid the discontinuity at
±pi, which is a genuine and commonly-hit trap.

### ActionParser

`LookupTableAction`: the Cartesian product of
`throttle {-1, 0, 1}` x `steer {-1, -0.5, 0, 0.5, 1}` x `boost {0, 1}` = **30 discrete
actions**. `get_action_space(agent)` returns `30`.

`RepeatAction(LookupTableAction(), repeats=4)` wraps it.

Discrete keeps the PPO policy a clean categorical distribution and mirrors the real
`LookupTableAction`.

### RewardFunction

`CombinedReward` with a deliberate magnitude ladder, matching the docs' pattern:

| Reward | Weight | Type |
|---|---|---|
| `SpeedTowardBallReward` | 0.01 | dense |
| `VelocityBallToGoalReward` | 0.1 | dense |
| `TouchReward` | 0.5 | event |
| `GoalReward` | 10.0 | terminal |

Dense terms are training wheels and must stay small enough that farming them can never
beat scoring. These weights are the subject of the Stage 3 reward-hacking experiment.

### DoneConditions

- **Termination:** `GoalCondition` — a goal was scored. The episode genuinely ended.
- **Truncation:** `AnyCondition(TimeoutCondition(30s), NoTouchTimeoutCondition(10s))`.

Both return `Dict[AgentID, bool]` from
`is_done(agents, state, shared_info)` — matching the real base class, *not* the
`is_done(agent, state) -> bool` signature shown in the official docs, which is wrong.

## PPO (`ppo.py`)

Written from scratch, ~150 lines, so the mechanics stay visible.

- Separate policy and value MLPs, 2 hidden layers of 256, tanh.
- Categorical policy over the 30 discrete actions.
- `gamma` 0.99, GAE `lambda` 0.95, clip 0.2, entropy coef 0.01, lr 3e-4.
- 8 in-process parallel envs, 256 steps each per iteration => 2048-step batch.
- 4 epochs per iteration, minibatch 256, advantages normalized per batch.

**The critical detail:** on `truncated`, the value target bootstraps with
`V(s_final)`. On `terminated`, it uses 0. Conflating these is the headline bug this
project exists to make concrete, and it gets a dedicated unit test.

## Dashboard (`train.py`, `render.py`)

Live matplotlib figure, two panels, updated every iteration:

- **Left:** the arena — walls, goals, ball, car with a heading indicator — replaying an
  episode sampled from the current policy.
- **Right:** mean episode reward vs iteration, plus touch rate and goal rate.

An episode is re-rendered every N iterations so behaviour change is directly observable
alongside the curve.

## Testing

`pytest`, split in two.

**Physics and integration** (guards correctness):
- A ball at rest stays at rest; a moving ball slows monotonically.
- A ball driven into a wall reverses with reduced speed.
- A car at full throttle accelerates along its yaw and plateaus at `CAR_MAX_SPEED`.
- A ball crossing the goal mouth sets `scoring_team`; crossing the back wall outside the
  mouth does not.
- `env.reset()` returns `Dict[AgentID, obs]`; `env.step()` returns a **4-tuple**.
- 1000 random steps with resets run without error, and no state field becomes NaN.
- GAE with a truncated final step bootstraps; with a terminated final step it does not.

**Config-object contracts** (these double as the Stage 3 exercise spec):
- Observation vector has shape (16,) and stays within [-2, 2] across randomized states.
- Each reward component has the expected sign in a hand-constructed state.
- Every `DoneCondition` returns a dict keyed by every agent in `agents`.

## Stage 3: the exercise

Once the reference implementation trains successfully, `exercises/` receives stubbed
copies of `obs.py`, `rewards.py`, and `dones.py` with the method bodies removed and the
contract tests pointed at them. The learner reimplements the three from understanding;
`pytest` grades it. The working `minirl/` versions remain available as reference.

Then the deliberate-breakage experiments, each a one-line config change with a predicted
and then observed outcome:

1. Raise a dense reward weight to 1.0 and watch the agent farm the proxy while its
   reward curve climbs.
2. Report goals as truncations rather than terminations, and watch value learning degrade.
3. Change `ACTION_REPEAT` and observe the effect on sample efficiency.
4. Remove the relative-position fields from the observation and watch learning slow.

## Stage 4: transfer

Default: add an opponent by changing `FixedTeamSizeMutator(blue_size=1, orange_size=1)`
and training via self-play. Because every object is already keyed by `AgentID`, only the
mutator and the training loop's action-dict handling change.

Alternative, if the learner names a target domain: replace `state.py` and `engine.py`
with that domain and keep the other five objects' structure. The spec fixes the swap
points — `engine.py` and `state.py` are the only modules that reference physics, and this
is enforced by the file layout.

## Success criteria

On CPU, within roughly 5 minutes of training:

1. Mean episode reward rises clearly above its initial value.
2. Touch rate rises from near 0 to above 80%.
3. Goals occur at a nonzero and rising rate.
4. Rendered behaviour visibly changes from random spinning to driving at the ball.

If (1) and (2) do not hold, the setup is considered broken and gets debugged before
proceeding — a flat curve is a bug, not a hyperparameter problem, at this scale.

## Environment setup

- Project venv at `/Users/rohan/Projects/RL Gyms/.venv` (approved). System Python untouched.
- Dependencies: `rlgym` (2.0.1, pure Python, zero deps), `numpy`, `torch` (~200MB),
  `matplotlib`, `pytest`.
- Verified: `rlgym` 2.0.1 imports cleanly on Python 3.13.0; `torch` 2.13.0 available.

## Risks

- **Physics instability.** Circle-circle impulses can inject energy and send the ball
  oscillating. Mitigated by clamping speeds, resolving overlap by position correction
  before applying impulse, and a monotonic-decay test.
- **Training too slow to feel live.** Mitigated by 15 Hz decisions, a 16-dim observation,
  a small network, and 30-second episode caps. If it still drags, reduce to 4 envs and
  shrink the arena.
- **Dashboard blocking the training loop.** Rendering every iteration will dominate
  runtime. Mitigated by drawing the curve every iteration but re-rendering an episode
  only every N.
