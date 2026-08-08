# DispatchSim — Stage 4: a maintenance-dispatch domain on the same six objects

**Date:** 2026-08-08
**Status:** Draft, awaiting review
**Depends on:** MiniRL (merged to master, 103 tests passing)

## Purpose

Stage 4 retires the Rocket League costume and tests the project's central claim —
that `core/` is domain-agnostic — by standing up a second domain beside
`domains/minirl/`: an apartment-building maintenance-dispatch simulator.

It does triple duty:

1. **Learning:** the learner sees the six objects wrap a business process instead of
   a game, which is the transferable skill this whole project exists to teach.
2. **Proof:** `core/` must run the new domain with only the generalization described
   below — and MiniRL must keep training bit-identically afterwards.
3. **Demo asset:** the same simulator doubles as an *evaluation harness* — run any
   agent (scripted rules today, an LLM agent later) through seeded scenarios and
   produce a failure report. This is the artifact used to pitch simulation work to
   agent companies: "here's a fake building, here's an agent inside it, here's
   exactly where it broke."

## The finding that shapes this spec: core/ is 90% agnostic

The import boundary is clean, but three attribute/vocabulary leaks were found by
inspection before writing any Stage 4 code:

| Leak | Where | Consequence |
|---|---|---|
| `env.state.scoring_team`, `env.state.last_touch_tick` read directly | `core/rollout.py:52-53` | `AttributeError` on any non-MiniRL state |
| `touched` / `scored` parameters, `touch_rate` / `goal_rate` keys | `core/metrics.py` | Metric names are MiniRL vocabulary |
| `"touch rate"` / `"goal rate"` labels and stored lists | `core/dashboard.py` | Same |

**Lesson worth stating in the code:** the boundary test enforced *imports*, not
*attribute coupling*. A module can be import-clean and still know another module's
field names.

### The fix: domain-supplied event extraction

`VecEnv` gains an `event_fn` argument — a callable the DOMAIN supplies:

```python
def event_fn(state) -> Dict[str, bool]:
    """Named episode-level events, read from live state before any auto-reset."""
```

- MiniRL's: `{"touch": state.last_touch_tick is not None, "goal": state.scoring_team is not None}`
- Dispatch's: `{"sla_clean": ..., "catastrophe": state.catastrophe is not None}`

`EpisodeTracker` and `Dashboard` become name-agnostic: the tracker records
`Dict[str, bool]` events and reports `<name>_rate` for each; the dashboard plots
whatever rate keys the snapshot contains. `train.py` wires MiniRL's `event_fn` in,
so all existing metric names survive unchanged.

**Regression gate (non-negotiable):** after this refactor, MiniRL's seed-0
250-iteration run must reproduce today's verified numbers exactly
(reward 1.133, touch 0.79, goal 0.08 at iter 250). The refactor moves code, not
behaviour.

## The domain: DispatchSim

You manage maintenance for one apartment building. Tickets arrive stochastically;
you assign vendors, defer, or watch things get worse. Severe tickets left
unattended can escalate into a catastrophe (burst pipe floods the building) that
ends the episode.

### Time

| Constant | Value |
|---|---|
| 1 tick | 1 simulated hour |
| `ACTION_REPEAT` | 1 (one decision per hour) |
| Episode length | 168 ticks = one week → **truncation** |
| Catastrophe | → **termination** |

The terminated/truncated mapping is the same lesson as MiniRL, in business clothes:
the week ending is not the building ceasing to exist — the critic must value the
ongoing state (open tickets carry into "next week" conceptually), while a
catastrophe genuinely ends the episode with nothing to bootstrap.

### State (`domains/dispatch/state.py`)

```python
@dataclass
class Ticket:
    kind: int            # 0 plumbing, 1 electrical, 2 HVAC, 3 appliance
    severity: int        # 1..5
    age_hours: int
    assigned_vendor: Optional[int]
    hours_to_fix: int    # remaining work once a vendor is on it

@dataclass
class Vendor:
    kind: int            # specialty: fixes this kind 2x faster; can work ANY kind.
                         # (Refined from "which kind they can fix": with 3 vendors
                         # and 4 kinds, hard matching leaves one kind unfixable —
                         # an unwinnable state. Specialists-are-faster keeps the
                         # matching decision interesting without it.)
    cost_per_job: float
    available: bool      # busy until current job finishes
    no_show_prob: float

@dataclass
class DispatchState:
    tickets: List[Ticket]        # open tickets, unbounded internally
    vendors: List[Vendor]        # V = 3
    budget_spent: float
    ticks: int
    resolved_in_sla: int
    resolved_late: int
    catastrophe: Optional[int]   # ticket kind that blew up, None otherwise
```

### TransitionEngine (`engine.py`)

Per tick:
1. New tickets arrive (per-hour Poisson-ish probability, kind and severity drawn
   from fixed tables; arrival rate higher during "business hours").
2. Assigned vendors progress their job; a no-show roll may waste the hour.
   Finished job → ticket resolved, SLA judged against `age_hours`.
3. Unattended tickets age. Severity-4+ tickets unattended past a threshold roll a
   per-hour catastrophe probability → sets `state.catastrophe`, episode terminates.
4. Deterministic given the env's seeded RNG (owned by the engine, seedable the
   same way `KickoffMutator`'s is).

### StateMutator (`mutators.py`)

`WeekStartMutator(rng)` — fresh week: 3–8 random open tickets (the backlog), all
vendors free, budget 0. This is the curriculum hook: a later `HeatWaveMutator`
(HVAC arrival rate ×4) or `VendorShortageMutator` slots into a `MutatorSequence`
without touching anything else — same trick as `KickoffMutator`.

### ObsBuilder (`obs.py`)

Fixed-size vector; the agent sees the **top K = 5 tickets** ranked by
(severity, age):

| Fields | Dims |
|---|---|
| per ticket slot: severity/5, age/72h, kind one-hot(4), assigned flag | 7 × 5 = 35 |
| per vendor: available, cost (normalized), kind one-hot(4) | 6 × 3 = 18 |
| budget spent (normalized), hour-of-week as (cos, sin) | 3 |
| **total** | **56** |

Empty slots zero-padded. The (cos, sin) hour encoding is the same
angle-discontinuity lesson as MiniRL's yaw.

### ActionParser (`actions.py`)

Discrete lookup table: assign (slot k, vendor v) = 15 actions, plus `defer` = **16
actions**. Invalid picks (empty slot, busy vendor, kind mismatch) are legal no-ops —
the agent must *learn* not to waste hours on them, which is itself signal.

### RewardFunction (`rewards.py`) — the ladder again

| Component | Weight | Type |
|---|---|---|
| `OpenTicketBurden` (−Σ severity/5 per open ticket) | 0.01 | dense penalty |
| `CostPenalty` (per $ spent) | 0.05 | event |
| `SlaResolutionReward` (+1 per in-SLA fix, +0.3 late) | 0.5 | event |
| `CatastropheReward` (−1 on catastrophe) | 10.0 | terminal |

Same three-orders-of-magnitude shape as MiniRL. The built-in reward-hacking
experiment: reward *ticket closure count* instead of SLA quality and watch the
agent learn to spam cheap vendors at low-severity tickets while emergencies burn —
Goodhart with a work belt.

### DoneConditions (`dones.py`)

`CatastropheCondition` → termination. `WeekEndCondition(168)` → truncation.

### Renderer (`render.py`)

`draw_board(ax, state)`: ticket queue as a column of severity-colored bars (aging
= darkening), vendors as status dots, budget bar, red flash on catastrophe.
Plugs into `core/dashboard.py` unchanged via `draw_fn`.

## Evaluation-harness mode (the demo asset)

`evaluate.py` — runs **any policy** through M seeded scenarios, no learning:

```python
def evaluate(agent_fn, n_scenarios=200, seed=0) -> report
# agent_fn: (obs: np.ndarray) -> int    — trained net, scripted rules, anything
```

Ships with two scripted baselines (`random`, `fifo` — always assign oldest ticket
to first free matching vendor) and produces a table: catastrophe rate, SLA rate,
cost/week, per-scenario worst cases with seeds so any failure replays exactly.

This file is the pitch artifact. An LLM-backed `agent_fn` is a later drop-in — the
interface already fits — but is explicitly out of scope now.

## Non-goals

- Real ticket text / NLP — tickets are structs, not language.
- Multi-building portfolios, real vendor data, calendar UIs.
- LLM integration (interface reserved, not built).
- Calibrated economics — numbers are plausible, not researched.
- Self-play / multi-agent. The 3 parked MiniRL findings (TouchReward attribution,
  opponent-blind obs, sustained-contact impulse) stay parked; they gate a 1v1
  branch we are deliberately not taking.

## Success criteria

1. **Reuse proof:** `domains/dispatch/` + `evaluate.py` + a thin `train.py` wiring
   change; `core/` diff is the event-extraction refactor only.
2. **Regression:** MiniRL seed-0 numbers reproduce exactly post-refactor.
3. **Learning:** trained policy beats `fifo` on catastrophe rate and beats `random`
   on everything, within ~10 min CPU.
4. **Harness:** `evaluate.py` report reproduces any listed failure from its seed.
5. All existing tests keep passing; new domain gets the same test discipline
   (physics-free, so: engine transitions, escalation, SLA accounting, obs bounds,
   action legality, done wiring — plus falsification-style tests for the ladder).

## Risks

- **The refactor regresses MiniRL silently.** Mitigated by the bit-identical
  seed-0 gate before any dispatch code lands.
- **16-action space too coarse to be interesting.** If the policy converges to
  trivial rules instantly, raise K/V — but only after baselines are measured.
- **Reward numbers need tuning to make `fifo` beatable but not unbeatable.**
  Expected; budgeted as an explicit tuning pass with the dashboard, which is
  itself a teaching moment.
