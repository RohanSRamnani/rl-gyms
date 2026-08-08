# DispatchSim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `core/` to be genuinely domain-agnostic, then build a maintenance-dispatch domain (`domains/dispatch/`) plus an evaluation harness (`evaluate.py`) on the unchanged core.

**Architecture:** `core/` loses its MiniRL vocabulary via a domain-supplied `event_fn(state) -> Dict[str, bool]`. `domains/dispatch/` implements the six RLGym objects for a one-building maintenance simulator (1 tick = 1 hour, week-long episodes). `train.py` gains a `--domain` flag; `evaluate.py` runs any `(obs) -> action` agent through seeded scenarios and reports failures.

**Tech Stack:** Python 3.13, `rlgym` 2.0.1, numpy, torch, matplotlib, pytest — all already in `./.venv`.

## Global Constraints

- All commands run from `/Users/rohan/Projects/RL Gyms` using `./.venv/bin/python` / `./.venv/bin/pytest`. Never pip install.
- `core/` must not import from `domains/` (enforced by `tests/test_boundary.py`) **and must not reference domain attribute names** — that is the leak this plan removes.
- **Regression gate:** after Task 1, `./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0` must end with exactly
  `iter   12 | reward    0.172 | touch  0.13 | goal  0.00 | entropy 3.223 | episodes 100`.
  Any deviation means the refactor changed behaviour, not just structure — a bug, full stop.
- Dispatch domain: 1 tick = 1 hour, no action repeat; `WEEK_TICKS = 168` truncates; catastrophe terminates. Obs is 56 dims; action space is 16.
- The obs builder and the engine must resolve "slot k" through the same `top_slots()` helper — a divergence silently breaks the action space.
- Class/file naming mirrors `domains/minirl/` file-for-file.
- Every task ends with a commit. Commit messages end with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

### Task 1: Generalize core/ — the event_fn refactor

**Files:**
- Modify: `core/metrics.py`, `core/rollout.py`, `core/dashboard.py`, `train.py`
- Modify: `domains/minirl/env.py` (add `episode_events`)
- Modify: `tests/test_rollout.py`, `tests/test_render.py`
- Test: `tests/test_metrics_generic.py`

**Interfaces:**
- Consumes: existing core modules.
- Produces:
  - `EpisodeTracker(event_names: Sequence[str] = (), window: int = 100)` with `record(reward, terminated, truncated, events: Dict[str, bool])` and `snapshot()` returning `mean_episode_reward`, `episodes`, and `f"{name}_rate"` per event name.
  - `VecEnv(env_fn, n_envs, agent_id=0, event_fn=None)` where `event_fn(state) -> Dict[str, bool]`; new method `episode_events() -> List[Dict[str, bool]]` returning pre-reset snapshots. `scoring_teams()` and `touched_flags()` are deleted.
  - `Dashboard` stores `rates: Dict[str, List[float]]` keyed by event name (replaces `touch_rates`/`goal_rates`) and plots every `*_rate` key in the snapshot.
  - `domains/minirl/env.py` exports `episode_events(state) -> {"touch": bool, "goal": bool}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics_generic.py
"""core/ must carry NO domain vocabulary. These tests use invented event names."""
import numpy as np

from core.metrics import EpisodeTracker


def test_tracker_reports_a_rate_per_event_name():
    tracker = EpisodeTracker(("frobnicated", "quuxed"))
    tracker.record(1.0, terminated=False, truncated=False,
                   events={"frobnicated": True, "quuxed": False})
    tracker.record(2.0, terminated=True, truncated=False,
                   events={"frobnicated": False, "quuxed": False})
    snap = tracker.snapshot()
    assert snap["episodes"] == 1
    assert np.isclose(snap["mean_episode_reward"], 3.0)
    assert np.isclose(snap["frobnicated_rate"], 1.0)   # OR across the episode
    assert np.isclose(snap["quuxed_rate"], 0.0)


def test_tracker_event_flags_reset_between_episodes():
    tracker = EpisodeTracker(("hit",))
    tracker.record(0.0, True, False, {"hit": True})     # episode 1: hit
    tracker.record(0.0, True, False, {"hit": False})    # episode 2: clean
    assert np.isclose(tracker.snapshot()["hit_rate"], 0.5)


def test_tracker_with_no_event_names_is_valid():
    tracker = EpisodeTracker()
    tracker.record(1.0, True, False, {})
    snap = tracker.snapshot()
    assert snap["episodes"] == 1
    assert set(snap) == {"mean_episode_reward", "episodes"}


def test_core_modules_contain_no_minirl_vocabulary():
    import pathlib
    for name in ("metrics.py", "rollout.py", "dashboard.py"):
        text = (pathlib.Path("core") / name).read_text().lower()
        for word in ("touch", "goal", "scoring_team", "ball", "car"):
            assert word not in text, f"core/{name} still says '{word}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_metrics_generic.py -v`
Expected: FAIL (TypeError on the new `EpisodeTracker` signature, and vocabulary assertions fail)

- [ ] **Step 3: Rewrite `core/metrics.py`**

```python
# core/metrics.py
"""Per-episode metric tracking. Domain-agnostic: event names come from the caller."""
from collections import deque
from typing import Dict, Sequence


class EpisodeTracker:
    """Accumulates per-step signals into per-episode statistics.

    `event_names` are domain-chosen (e.g. ("touch", "goal") or
    ("catastrophe", "sla_clean")). Each event is OR-accumulated across an
    episode and reported as `<name>_rate` over the last `window` episodes.
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
```

- [ ] **Step 4: Rewrite the event plumbing in `core/rollout.py`**

Replace the `__init__` event fields, the pre-reset snapshot, the two deleted methods, and the `collect_rollout` metric block:

```python
# core/rollout.py  — full replacement for the affected parts
"""Vectorized rollout collection. Domain-agnostic: takes an env factory and an
event extractor; never references a domain's state fields by name."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.metrics import EpisodeTracker
from core.ppo import ActorCritic


class VecEnv:
    """Runs N single-agent RLGym environments in-process with auto-reset.

    `event_fn(state) -> Dict[str, bool]` is supplied by the DOMAIN and read
    inside `step` BEFORE any auto-reset, because the reset wipes the evidence
    (a mutator clearing an episode field is what previously pinned a metric to
    zero forever). `final_obs[i]` is likewise the pre-reset observation, which
    PPO needs to bootstrap truncated episodes.
    """

    def __init__(self, env_fn: Callable[[], Any], n_envs: int, agent_id: int = 0,
                 event_fn: Optional[Callable[[Any], Dict[str, bool]]] = None) -> None:
        self.envs = [env_fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        self.agent_id = agent_id
        self.event_fn = event_fn
        self.last_obs: Optional[np.ndarray] = None   # carried across rollouts
        self._episode_events: List[Dict[str, bool]] = [{} for _ in range(n_envs)]

    def reset(self) -> np.ndarray:
        return np.stack([env.reset()[self.agent_id] for env in self.envs])

    def step(self, actions: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Optional[np.ndarray]]
    ]:
        observations, rewards = [], []
        terminated_flags, truncated_flags = [], []
        final_observations: List[Optional[np.ndarray]] = []

        for index, env in enumerate(self.envs):
            obs, reward, terminated, truncated = env.step(
                {self.agent_id: int(actions[index])}
            )
            done_terminated = bool(terminated[self.agent_id])
            done_truncated = bool(truncated[self.agent_id])

            rewards.append(float(reward[self.agent_id]))
            terminated_flags.append(done_terminated)
            truncated_flags.append(done_truncated)

            # Snapshot before the reset destroys the evidence.
            self._episode_events[index] = (
                self.event_fn(env.state) if self.event_fn is not None else {}
            )

            if done_terminated or done_truncated:
                final_observations.append(obs[self.agent_id])
                observations.append(env.reset()[self.agent_id])
            else:
                final_observations.append(None)
                observations.append(obs[self.agent_id])

        return (np.stack(observations), np.array(rewards, dtype=np.float32),
                np.array(terminated_flags), np.array(truncated_flags),
                final_observations)

    def episode_events(self) -> List[Dict[str, bool]]:
        """Event flags as of the last `step`, captured before any auto-reset."""
        return [dict(events) for events in self._episode_events]
```

In `collect_rollout`, replace the old flag reads and `tracker.record` call:

```python
        next_obs, rewards, terminated, truncated, final_obs = vec_env.step(
            actions_buffer[t]
        )
        # Snapshots taken inside step(), before the auto-reset.
        events = vec_env.episode_events()
```

```python
        for i in range(n):
            tracker.record(float(rewards[i]), bool(terminated[i]), bool(truncated[i]),
                           events[i])
```

The bootstrap logic is untouched. Do not modify anything else in the file.

- [ ] **Step 5: Generalize `core/dashboard.py`**

Replace the hardcoded rate lists and plotting:

```python
# in Dashboard.__init__ — replace touch_rates/goal_rates with:
        self.rates: Dict[str, List[float]] = {}
```

```python
# Dashboard.update_curve — full replacement
    def update_curve(self, iteration: int, snapshot: Dict[str, float]) -> None:
        self.iterations.append(iteration)
        self.rewards.append(snapshot["mean_episode_reward"])
        rate_keys = sorted(k for k in snapshot if k.endswith("_rate"))
        for key in rate_keys:
            self.rates.setdefault(key[:-5], []).append(snapshot[key])

        self.curve_ax.clear()
        self.rate_ax.clear()
        self.curve_ax.plot(self.iterations, self.rewards, color="tab:blue",
                           label="mean episode reward")
        self.curve_ax.set_xlabel("iteration")
        self.curve_ax.set_ylabel("mean episode reward", color="tab:blue")
        self.curve_ax.grid(alpha=0.3)

        styles = ["--", ":", "-.", (0, (1, 1))]
        colours = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
        for i, (name, series) in enumerate(sorted(self.rates.items())):
            self.rate_ax.plot(self.iterations[-len(series):], series,
                              color=colours[i % len(colours)],
                              linestyle=styles[i % len(styles)],
                              label=f"{name} rate")
        self.rate_ax.set_ylabel("rate")
        self.rate_ax.set_ylim(-0.05, 1.05)

        handles = self.curve_ax.get_legend_handles_labels()[0] + \
            self.rate_ax.get_legend_handles_labels()[0]
        labels = self.curve_ax.get_legend_handles_labels()[1] + \
            self.rate_ax.get_legend_handles_labels()[1]
        self.curve_ax.legend(handles, labels, loc="upper left", fontsize=9)
        self._flush()
```

`Dict` and `List` are already imported in the file's `typing` import; verify.

- [ ] **Step 6: Add MiniRL's event extractor and wire `train.py`**

Append to `domains/minirl/env.py` (and add `"episode_events"` to `__all__`):

```python
def episode_events(state) -> dict:
    """MiniRL's episode-level events, for core.rollout.VecEnv(event_fn=...)."""
    return {
        "touch": state.last_touch_tick is not None,
        "goal": state.scoring_team is not None,
    }
```

In `train.py`: import it, construct `EpisodeTracker(("touch", "goal"))`, and pass
`event_fn=episode_events` into `VecEnv(...)`. The print line already reads
`snapshot['touch_rate']` / `snapshot['goal_rate']`, which the tracker still
produces — the console format must not change (the regression gate diffs it).

- [ ] **Step 7: Update the affected existing tests**

In `tests/test_rollout.py`: construct `EpisodeTracker()` (bootstrap tests) or
`EpisodeTracker(("touch", "goal"))` with `VecEnv(build_env, n_envs=1, event_fn=episode_events)`
(the goal-rate and final-step-touch tests); import `episode_events` from
`domains.minirl.env`; update `test_episode_tracker_averages_completed_episodes` to
the events-dict signature. In `tests/test_render.py`: the two dashboard tests use
`dashboard.rates["touch"]` / `dashboard.rates["goal"]` instead of
`dashboard.touch_rates` / `dashboard.goal_rates`.

- [ ] **Step 8: Run the full suite**

Run: `./.venv/bin/pytest tests/ -v`
Expected: **107 passed, 3 skipped** (103 + 4 new)

- [ ] **Step 9: THE REGRESSION GATE**

Run: `./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0 | tail -1`
Expected — byte-for-byte:
`iter   12 | reward    0.172 | touch  0.13 | goal  0.00 | entropy 3.223 | episodes 100`

If it differs in ANY digit, the refactor changed behaviour. Debug; do not proceed, do not re-tune.

- [ ] **Step 10: Commit**

```bash
git add core/ domains/minirl/env.py train.py tests/
git commit -m "refactor: core/ event plumbing is domain-supplied, not hardcoded

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Dispatch state, constants, and the shared slot ordering

**Files:**
- Create: `domains/dispatch/__init__.py` (empty), `domains/dispatch/state.py`
- Test: `tests/test_dispatch_state.py`

**Interfaces:**
- Produces: dataclasses `Ticket`, `Vendor`, `DispatchState`; helper `top_slots(state, k=K_SLOTS) -> List[int]`; constants below. `top_slots` is THE slot authority — obs and engine both use it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_state.py
import numpy as np
from domains.dispatch.state import (
    Ticket, Vendor, DispatchState, top_slots,
    N_TICKET_KINDS, K_SLOTS, N_VENDORS, WEEK_TICKS, SLA_HOURS, MAX_SEVERITY,
    ESCALATION_SEVERITY, ESCALATION_GRACE_HOURS, CATASTROPHE_PROB,
    ARRIVAL_PROB_DAY, ARRIVAL_PROB_NIGHT, FIX_HOURS_MIN, FIX_HOURS_MAX,
    MISMATCH_FIX_MULTIPLIER, VENDOR_COSTS, VENDOR_NO_SHOW_PROBS, VENDOR_KINDS,
    SEVERITY_PROBS, MAX_AGE_NORM, MAX_BUDGET_NORM,
)


def make_ticket(tid=0, kind=0, severity=3, age=0):
    return Ticket(ticket_id=tid, kind=kind, severity=severity, age_hours=age)


def test_constants_are_coherent():
    assert len(VENDOR_COSTS) == len(VENDOR_NO_SHOW_PROBS) == len(VENDOR_KINDS) == N_VENDORS
    assert ESCALATION_SEVERITY <= MAX_SEVERITY
    assert 0 < CATASTROPHE_PROB < 1
    assert abs(sum(SEVERITY_PROBS) - 1.0) < 1e-9
    assert len(SEVERITY_PROBS) == MAX_SEVERITY
    assert WEEK_TICKS == 168 and K_SLOTS == 5
    assert MISMATCH_FIX_MULTIPLIER == 2


def test_state_constructs_with_defaults():
    state = DispatchState(tickets=[], vendors=[])
    assert state.ticks == 0 and state.budget_spent == 0.0
    assert state.catastrophe is None
    assert state.resolved_this_tick == [] and state.spent_this_tick == 0.0


def test_top_slots_orders_by_severity_then_age_then_id():
    state = DispatchState(tickets=[
        make_ticket(tid=0, severity=2, age=50),
        make_ticket(tid=1, severity=5, age=1),
        make_ticket(tid=2, severity=5, age=9),
        make_ticket(tid=3, severity=2, age=50),
    ], vendors=[])
    order = top_slots(state)
    assert order[:2] == [2, 1]          # both sev 5, older first
    assert order[2:] == [0, 3]          # sev 2 tie on age -> lower id first


def test_top_slots_caps_at_k():
    state = DispatchState(tickets=[make_ticket(tid=i) for i in range(9)], vendors=[])
    assert len(top_slots(state)) == K_SLOTS


def test_vendor_defaults_to_free():
    vendor = Vendor(kind=0, cost_per_job=100.0, no_show_prob=0.1)
    assert vendor.busy_ticket_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'domains.dispatch'`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/state.py
"""State types and tuning constants for DispatchSim.

1 tick = 1 simulated hour. All money in dollars, all ages in hours.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- Scale -------------------------------------------------------------------
N_TICKET_KINDS = 4                     # plumbing, electrical, hvac, appliance
KIND_NAMES = ("plumbing", "electrical", "hvac", "appliance")
K_SLOTS = 5                            # tickets visible to the agent
N_VENDORS = 3
WEEK_TICKS = 168                       # one episode = one week of hours

# --- Service levels ----------------------------------------------------------
SLA_HOURS = 24
MAX_SEVERITY = 5
SEVERITY_PROBS = (0.30, 0.30, 0.20, 0.15, 0.05)   # arrival severity 1..5

# --- Escalation --------------------------------------------------------------
ESCALATION_SEVERITY = 4                # >= this can blow up if ignored
ESCALATION_GRACE_HOURS = 12            # unattended hours before rolls start
CATASTROPHE_PROB = 0.05                # per hour, per qualifying ticket

# --- Arrivals ----------------------------------------------------------------
ARRIVAL_PROB_DAY = 0.35                # new-ticket prob per hour, 08:00-20:00
ARRIVAL_PROB_NIGHT = 0.10
DAY_START_HOUR, DAY_END_HOUR = 8, 20

# --- Vendors -----------------------------------------------------------------
FIX_HOURS_MIN, FIX_HOURS_MAX = 2, 6    # base job length (specialty work)
MISMATCH_FIX_MULTIPLIER = 2            # off-specialty jobs take twice as long
VENDOR_KINDS = (0, 1, 2)               # specialties (no appliance specialist)
VENDOR_COSTS = (150.0, 100.0, 60.0)    # cheaper vendor = flakier
VENDOR_NO_SHOW_PROBS = (0.05, 0.15, 0.30)

# --- Observation normalizers -------------------------------------------------
MAX_AGE_NORM = 72.0
MAX_BUDGET_NORM = 5000.0
MAX_COST_NORM = max(VENDOR_COSTS)


@dataclass
class Ticket:
    ticket_id: int
    kind: int
    severity: int
    age_hours: int = 0
    assigned_vendor: Optional[int] = None
    hours_to_fix: int = 0              # remaining work once assigned


@dataclass
class Vendor:
    kind: int                          # specialty: 2x faster on this kind
    cost_per_job: float
    no_show_prob: float
    busy_ticket_id: Optional[int] = None

    @property
    def available(self) -> bool:
        return self.busy_ticket_id is None


@dataclass
class DispatchState:
    tickets: List[Ticket]
    vendors: List[Vendor]
    budget_spent: float = 0.0
    ticks: int = 0
    next_ticket_id: int = 0
    resolved_in_sla: int = 0
    resolved_late: int = 0
    # Per-tick event channels, cleared at the start of every engine tick.
    resolved_this_tick: List[Tuple[int, bool]] = field(default_factory=list)  # (severity, in_sla)
    spent_this_tick: float = 0.0
    catastrophe: Optional[int] = None  # kind that blew up, None otherwise


def top_slots(state: DispatchState, k: int = K_SLOTS) -> List[int]:
    """Indices of the top-k tickets, ranked (severity desc, age desc, id asc).

    THE single slot authority: the obs builder shows these slots and the engine
    resolves 'assign slot k' through this same function. If they ever diverge,
    the agent acts on a different ticket than the one it saw.
    """
    order = sorted(range(len(state.tickets)),
                   key=lambda i: (-state.tickets[i].severity,
                                  -state.tickets[i].age_hours,
                                  state.tickets[i].ticket_id))
    return order[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_state.py -v`
Expected: 5 passed. Full suite: 112 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch tests/test_dispatch_state.py
git commit -m "feat: add DispatchSim state, constants, and slot ordering

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: DispatchEngine

**Files:**
- Create: `domains/dispatch/engine.py`
- Test: `tests/test_dispatch_engine.py`

**Interfaces:**
- Consumes: everything from Task 2.
- Produces: `DispatchEngine(TransitionEngine[int, DispatchState, int])`, constructed with `rng: Optional[np.random.Generator]`. Engine action is a plain `int` in `[0, 16)`: `a == 15` defers; else `slot = a // N_VENDORS`, `vendor = a % N_VENDORS`. Invalid picks are silent no-ops.

**Tick order inside `step` (one call = one hour):** clear per-tick channels → apply action → vendors work (no-show roll, else progress; finish → resolve + SLA accounting) → open tickets age → unassigned severe tickets past grace roll catastrophe → arrivals → `ticks += 1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_engine.py
import numpy as np
from domains.dispatch.state import (
    DispatchState, Ticket, Vendor, top_slots,
    N_VENDORS, SLA_HOURS, ESCALATION_GRACE_HOURS, WEEK_TICKS,
    VENDOR_COSTS, MISMATCH_FIX_MULTIPLIER, FIX_HOURS_MAX,
)
from domains.dispatch.engine import DispatchEngine, DEFER_ACTION


def seeded_engine(tickets=None, seed=0, no_shows=False):
    engine = DispatchEngine(rng=np.random.default_rng(seed))
    state = engine.create_base_state()
    state.vendors = [Vendor(kind=k, cost_per_job=VENDOR_COSTS[k],
                            no_show_prob=(0.0 if not no_shows else 1.0))
                     for k in range(N_VENDORS)]
    for t in (tickets or []):
        state.tickets.append(t)
        state.next_ticket_id = max(state.next_ticket_id, t.ticket_id + 1)
    engine.set_state(state, {})
    return engine


def tk(tid, kind=0, severity=3, age=0):
    return Ticket(ticket_id=tid, kind=kind, severity=severity, age_hours=age)


def test_agents_and_max():
    engine = seeded_engine()
    assert engine.agents == [0] and engine.max_num_agents == 1


def test_assign_charges_cost_and_occupies_vendor():
    engine = seeded_engine([tk(0, kind=0)])
    engine.step({0: 0 * N_VENDORS + 0}, {})          # slot 0 -> vendor 0
    state = engine.state
    assert state.vendors[0].busy_ticket_id == 0
    assert state.tickets[0].assigned_vendor == 0
    assert state.budget_spent == VENDOR_COSTS[0]
    assert state.spent_this_tick == 0.0 or state.ticks == 1  # cleared next tick


def test_specialty_matters_for_fix_time():
    match = seeded_engine([tk(0, kind=1)])
    match.step({0: 0 * N_VENDORS + 1}, {})           # electrical -> electrician
    mismatch = seeded_engine([tk(0, kind=3)])
    mismatch.step({0: 0 * N_VENDORS + 1}, {})        # appliance -> electrician
    assert mismatch.state.tickets[0].hours_to_fix \
        == match.state.tickets[0].hours_to_fix * MISMATCH_FIX_MULTIPLIER


def test_job_completes_and_counts_sla():
    engine = seeded_engine([tk(0, kind=0, age=0)])
    engine.step({0: 0}, {})                          # assign slot0 -> vendor0
    for _ in range(FIX_HOURS_MAX * MISMATCH_FIX_MULTIPLIER):
        engine.step({0: DEFER_ACTION}, {})
    state = engine.state
    assert state.tickets == []                       # resolved and removed
    assert state.vendors[0].available
    assert state.resolved_in_sla == 1 and state.resolved_late == 0


def test_late_resolution_counts_as_late():
    engine = seeded_engine([tk(0, kind=0, age=SLA_HOURS + 1)])
    engine.step({0: 0}, {})
    for _ in range(FIX_HOURS_MAX * 2):
        engine.step({0: DEFER_ACTION}, {})
    assert engine.state.resolved_late == 1


def test_permanent_no_show_never_finishes():
    engine = seeded_engine([tk(0, kind=0)], no_shows=True)
    engine.step({0: 0}, {})
    for _ in range(50):
        engine.step({0: DEFER_ACTION}, {})
    assert engine.state.tickets and engine.state.tickets[0].assigned_vendor == 0


def test_invalid_actions_are_no_ops():
    engine = seeded_engine([tk(0)])
    before_budget = engine.state.budget_spent
    engine.step({0: 4 * N_VENDORS + 2}, {})          # slot 4 is empty
    assert engine.state.budget_spent == before_budget
    engine.step({0: 0}, {})                          # occupy vendor 0
    engine.step({0: 0}, {})                          # vendor 0 busy -> no-op
    assert engine.state.budget_spent == VENDOR_COSTS[0]


def test_unattended_severe_ticket_eventually_catastrophises():
    engine = seeded_engine([tk(0, severity=5, age=ESCALATION_GRACE_HOURS)])
    for _ in range(200):
        engine.step({0: DEFER_ACTION}, {})
        if engine.state.catastrophe is not None:
            break
    assert engine.state.catastrophe == 0             # the ticket's kind


def test_assigned_severe_ticket_never_catastrophises():
    engine = seeded_engine([tk(0, severity=5, age=ESCALATION_GRACE_HOURS)],
                           no_shows=True)            # assigned forever, never done
    engine.step({0: 0}, {})
    for _ in range(300):
        engine.step({0: DEFER_ACTION}, {})
    assert engine.state.catastrophe is None


def test_arrivals_happen_over_a_week():
    engine = seeded_engine()
    for _ in range(WEEK_TICKS):
        engine.step({0: DEFER_ACTION}, {})
    total = len(engine.state.tickets) + engine.state.resolved_in_sla \
        + engine.state.resolved_late
    assert total > 10


def test_same_seed_same_week():
    def run(seed):
        engine = seeded_engine(seed=seed)
        for _ in range(100):
            engine.step({0: DEFER_ACTION}, {})
        s = engine.state
        return (len(s.tickets), s.next_ticket_id,
                tuple((t.kind, t.severity) for t in s.tickets))
    assert run(7) == run(7)
    assert run(7) != run(8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/engine.py
"""The DispatchSim transition engine — the only module that advances the world."""
from typing import Any, Dict, List, Optional

import numpy as np
from rlgym.api import TransitionEngine

from domains.dispatch.state import (
    ARRIVAL_PROB_DAY, ARRIVAL_PROB_NIGHT, CATASTROPHE_PROB, DAY_END_HOUR,
    DAY_START_HOUR, ESCALATION_GRACE_HOURS, ESCALATION_SEVERITY, FIX_HOURS_MAX,
    FIX_HOURS_MIN, K_SLOTS, MISMATCH_FIX_MULTIPLIER, N_TICKET_KINDS, N_VENDORS,
    SEVERITY_PROBS, SLA_HOURS, DispatchState, Ticket, top_slots,
)

N_ACTIONS = K_SLOTS * N_VENDORS + 1
DEFER_ACTION = N_ACTIONS - 1


class DispatchEngine(TransitionEngine[int, DispatchState, int]):
    """One `step` = one simulated hour. Engine action is a plain int in [0, 16)."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self._state: DispatchState = self.create_base_state()
        self._config: Dict[str, Any] = {}

    @property
    def agents(self) -> List[int]:
        return [0]

    @property
    def max_num_agents(self) -> int:
        return 1

    @property
    def state(self) -> DispatchState:
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        self._config = value

    def create_base_state(self) -> DispatchState:
        return DispatchState(tickets=[], vendors=[])

    def set_state(self, desired_state: DispatchState, shared_info: Dict[str, Any]) -> DispatchState:
        self._state = desired_state
        return self._state

    def step(self, actions: Dict[int, int], shared_info: Dict[str, Any]) -> DispatchState:
        state = self._state
        state.resolved_this_tick = []
        state.spent_this_tick = 0.0

        self._apply_action(int(actions[0]))
        self._vendors_work()
        for ticket in state.tickets:
            ticket.age_hours += 1
        self._roll_escalations()
        self._arrivals()
        state.ticks += 1
        return state

    def close(self) -> None:
        pass

    # -- internals ------------------------------------------------------------

    def _apply_action(self, action: int) -> None:
        state = self._state
        if action == DEFER_ACTION:
            return
        slot, vendor_id = divmod(action, N_VENDORS)
        slots = top_slots(state)
        if slot >= len(slots):
            return                                   # empty slot: no-op
        ticket = state.tickets[slots[slot]]
        vendor = state.vendors[vendor_id]
        if not vendor.available or ticket.assigned_vendor is not None:
            return                                   # busy vendor / already worked
        base_hours = int(self.rng.integers(FIX_HOURS_MIN, FIX_HOURS_MAX + 1))
        multiplier = 1 if vendor.kind == ticket.kind else MISMATCH_FIX_MULTIPLIER
        ticket.assigned_vendor = vendor_id
        ticket.hours_to_fix = base_hours * multiplier
        vendor.busy_ticket_id = ticket.ticket_id
        state.budget_spent += vendor.cost_per_job    # charged upfront
        state.spent_this_tick += vendor.cost_per_job

    def _vendors_work(self) -> None:
        state = self._state
        by_id = {t.ticket_id: t for t in state.tickets}
        for vendor in state.vendors:
            if vendor.busy_ticket_id is None:
                continue
            if self.rng.random() < vendor.no_show_prob:
                continue                             # wasted hour
            ticket = by_id[vendor.busy_ticket_id]
            ticket.hours_to_fix -= 1
            if ticket.hours_to_fix <= 0:
                in_sla = ticket.age_hours <= SLA_HOURS
                if in_sla:
                    state.resolved_in_sla += 1
                else:
                    state.resolved_late += 1
                state.resolved_this_tick.append((ticket.severity, in_sla))
                state.tickets.remove(ticket)
                vendor.busy_ticket_id = None

    def _roll_escalations(self) -> None:
        state = self._state
        for ticket in state.tickets:
            if (ticket.assigned_vendor is None
                    and ticket.severity >= ESCALATION_SEVERITY
                    and ticket.age_hours >= ESCALATION_GRACE_HOURS
                    and self.rng.random() < CATASTROPHE_PROB):
                state.catastrophe = ticket.kind
                return

    def _arrivals(self) -> None:
        state = self._state
        hour = state.ticks % 24
        prob = ARRIVAL_PROB_DAY if DAY_START_HOUR <= hour < DAY_END_HOUR else ARRIVAL_PROB_NIGHT
        if self.rng.random() < prob:
            state.tickets.append(Ticket(
                ticket_id=state.next_ticket_id,
                kind=int(self.rng.integers(N_TICKET_KINDS)),
                severity=int(self.rng.choice(len(SEVERITY_PROBS), p=SEVERITY_PROBS)) + 1,
            ))
            state.next_ticket_id += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_engine.py -v`
Expected: 11 passed. Full suite: 123 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch/engine.py tests/test_dispatch_engine.py
git commit -m "feat: add DispatchEngine with arrivals, no-shows, and escalation

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Mutator and done conditions

**Files:**
- Create: `domains/dispatch/mutators.py`, `domains/dispatch/dones.py`
- Test: `tests/test_dispatch_mutators_dones.py`

**Interfaces:**
- Produces: `WeekStartMutator(rng=None)` (3–8 backlog tickets, fresh vendors from the constant tables, zeroed counters); `CatastropheCondition` (termination), `WeekEndCondition(week_ticks=WEEK_TICKS)` (truncation). Done conditions use the list-in/dict-out `is_done` signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_mutators_dones.py
import numpy as np
from domains.dispatch.state import DispatchState, N_VENDORS, WEEK_TICKS, VENDOR_KINDS
from domains.dispatch.mutators import WeekStartMutator
from domains.dispatch.dones import CatastropheCondition, WeekEndCondition


def blank():
    return DispatchState(tickets=[], vendors=[])


def test_week_start_populates_vendors_and_backlog():
    state = blank()
    WeekStartMutator(rng=np.random.default_rng(0)).apply(state, {})
    assert len(state.vendors) == N_VENDORS
    assert tuple(v.kind for v in state.vendors) == VENDOR_KINDS
    assert 3 <= len(state.tickets) <= 8
    assert all(v.available for v in state.vendors)
    assert state.ticks == 0 and state.catastrophe is None


def test_week_start_clears_a_dirty_state():
    state = blank()
    state.ticks, state.budget_spent, state.catastrophe = 99, 500.0, 2
    state.resolved_in_sla, state.resolved_late = 4, 2
    WeekStartMutator(rng=np.random.default_rng(1)).apply(state, {})
    assert state.ticks == 0 and state.budget_spent == 0.0
    assert state.catastrophe is None
    assert state.resolved_in_sla == 0 and state.resolved_late == 0


def test_backlogs_vary_between_weeks():
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(10):
        state = blank()
        WeekStartMutator(rng=rng).apply(state, {})
        seen.add(tuple((t.kind, t.severity) for t in state.tickets))
    assert len(seen) > 1


def test_catastrophe_condition():
    condition = CatastropheCondition()
    quiet, burst = blank(), blank()
    burst.catastrophe = 1
    condition.reset([0], quiet, {})
    assert condition.is_done([0], quiet, {}) == {0: False}
    assert condition.is_done([0], burst, {}) == {0: True}


def test_week_end_condition_fires_at_the_limit():
    condition = WeekEndCondition()
    condition.reset([0], blank(), {})
    early, done = blank(), blank()
    early.ticks, done.ticks = WEEK_TICKS - 1, WEEK_TICKS
    assert condition.is_done([0], early, {}) == {0: False}
    assert condition.is_done([0], done, {}) == {0: True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_mutators_dones.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/mutators.py
"""State mutators — the curriculum hook. A HeatWaveMutator slots in here later."""
from typing import Any, Dict, Optional

import numpy as np
from rlgym.api import StateMutator

from domains.dispatch.state import (
    N_TICKET_KINDS, SEVERITY_PROBS, VENDOR_COSTS, VENDOR_KINDS,
    VENDOR_NO_SHOW_PROBS, DispatchState, Ticket, Vendor,
)


class WeekStartMutator(StateMutator[DispatchState]):
    """Fresh week: standard vendor roster, a randomized 3-8 ticket backlog."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply(self, state: DispatchState, shared_info: Dict[str, Any]) -> None:
        state.vendors = [
            Vendor(kind=VENDOR_KINDS[i], cost_per_job=VENDOR_COSTS[i],
                   no_show_prob=VENDOR_NO_SHOW_PROBS[i])
            for i in range(len(VENDOR_KINDS))
        ]
        state.tickets = []
        backlog = int(self.rng.integers(3, 9))
        for _ in range(backlog):
            state.tickets.append(Ticket(
                ticket_id=state.next_ticket_id if False else len(state.tickets),
                kind=int(self.rng.integers(N_TICKET_KINDS)),
                severity=int(self.rng.choice(len(SEVERITY_PROBS), p=SEVERITY_PROBS)) + 1,
                age_hours=int(self.rng.integers(0, 12)),
            ))
        state.next_ticket_id = len(state.tickets)
        state.budget_spent = 0.0
        state.ticks = 0
        state.resolved_in_sla = 0
        state.resolved_late = 0
        state.resolved_this_tick = []
        state.spent_this_tick = 0.0
        state.catastrophe = None
```

Note: simplify that ternary — ticket ids in the backlog are `0..backlog-1` and
`next_ticket_id = backlog`. Write it plainly:
`ticket_id=len(state.tickets)`.

```python
# domains/dispatch/dones.py
"""CatastropheCondition TERMINATES (the building flooded; there is no next hour
worth valuing). WeekEndCondition TRUNCATES (the week ended, the building did not
— the critic must value the ongoing state)."""
from typing import Any, Dict, List

from rlgym.api import DoneCondition

from domains.dispatch.state import WEEK_TICKS, DispatchState


class CatastropheCondition(DoneCondition[int, DispatchState]):
    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: DispatchState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        burst = state.catastrophe is not None
        return {agent: burst for agent in agents}


class WeekEndCondition(DoneCondition[int, DispatchState]):
    def __init__(self, week_ticks: int = WEEK_TICKS) -> None:
        self.week_ticks = week_ticks

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: DispatchState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        over = state.ticks >= self.week_ticks
        return {agent: over for agent in agents}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_mutators_dones.py -v`
Expected: 5 passed. Full suite: 128 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch/mutators.py domains/dispatch/dones.py tests/test_dispatch_mutators_dones.py
git commit -m "feat: add DispatchSim week-start mutator and done conditions

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Observation builder

**Files:**
- Create: `domains/dispatch/obs.py`
- Test: `tests/test_dispatch_obs.py`

**Interfaces:**
- Produces: `DispatchObs(ObsBuilder[int, np.ndarray, DispatchState, int])`, `OBS_SIZE = 56`.
  Layout: 5 slots × [severity/5, age/72 clipped, kind one-hot(4), assigned] = 35;
  3 vendors × [available, cost/MAX_COST_NORM, kind one-hot(4)] = 18;
  budget/MAX_BUDGET_NORM clipped, cos(hour angle), sin(hour angle) = 3.
  Hour angle = `2π · (ticks % WEEK_TICKS) / WEEK_TICKS`. Slots come from `top_slots`; empty slots are all-zero.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_obs.py
import numpy as np
from domains.dispatch.state import (
    DispatchState, Ticket, Vendor, top_slots, K_SLOTS, MAX_AGE_NORM,
    VENDOR_COSTS, VENDOR_KINDS, VENDOR_NO_SHOW_PROBS, WEEK_TICKS,
)
from domains.dispatch.obs import DispatchObs, OBS_SIZE

SLOT_DIMS, VENDOR_DIMS = 7, 6


def make_state(tickets=(), ticks=0):
    vendors = [Vendor(kind=VENDOR_KINDS[i], cost_per_job=VENDOR_COSTS[i],
                      no_show_prob=VENDOR_NO_SHOW_PROBS[i]) for i in range(3)]
    return DispatchState(tickets=list(tickets), vendors=vendors, ticks=ticks)


def test_obs_space_and_shape():
    builder = DispatchObs()
    assert builder.get_obs_space(0) == OBS_SIZE == 56
    state = make_state()
    builder.reset([0], state, {})
    obs = builder.build_obs([0], state, {})[0]
    assert obs.shape == (56,) and obs.dtype == np.float32


def test_empty_slots_are_zero():
    builder = DispatchObs()
    obs = builder.build_obs([0], make_state(), {})[0]
    assert np.all(obs[:K_SLOTS * SLOT_DIMS] == 0.0)


def test_slot_zero_is_the_top_ticket():
    t_low = Ticket(ticket_id=0, kind=2, severity=1)
    t_high = Ticket(ticket_id=1, kind=3, severity=5, age_hours=36)
    state = make_state([t_low, t_high])
    assert top_slots(state)[0] == 1
    obs = DispatchObs().build_obs([0], state, {})[0]
    assert np.isclose(obs[0], 1.0)                       # severity 5/5
    assert np.isclose(obs[1], 36 / MAX_AGE_NORM)
    assert np.isclose(obs[2 + 3], 1.0)                   # kind one-hot index 3
    assert obs[6] == 0.0                                 # unassigned


def test_age_is_clipped_to_one():
    t = Ticket(ticket_id=0, kind=0, severity=3, age_hours=500)
    obs = DispatchObs().build_obs([0], make_state([t]), {})[0]
    assert np.isclose(obs[1], 1.0)


def test_hour_encoding_is_continuous_at_the_week_boundary():
    builder = DispatchObs()
    end = builder.build_obs([0], make_state(ticks=WEEK_TICKS - 1), {})[0]
    start = builder.build_obs([0], make_state(ticks=0), {})[0]
    assert abs(end[-2] - start[-2]) < 0.1 and abs(end[-1] - start[-1]) < 0.1


def test_obs_stays_bounded_on_random_states():
    rng = np.random.default_rng(0)
    builder = DispatchObs()
    for _ in range(200):
        tickets = [Ticket(ticket_id=i, kind=int(rng.integers(4)),
                          severity=int(rng.integers(1, 6)),
                          age_hours=int(rng.integers(0, 300)))
                   for i in range(int(rng.integers(0, 12)))]
        state = make_state(tickets, ticks=int(rng.integers(0, WEEK_TICKS)))
        state.budget_spent = float(rng.uniform(0, 20000))
        obs = builder.build_obs([0], state, {})[0]
        assert np.all(np.abs(obs) <= 1.0 + 1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_obs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/obs.py
"""Observation builder: top-K ticket slots + vendor roster + clock."""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ObsBuilder

from domains.dispatch.state import (
    K_SLOTS, MAX_AGE_NORM, MAX_BUDGET_NORM, MAX_COST_NORM, MAX_SEVERITY,
    N_TICKET_KINDS, WEEK_TICKS, DispatchState, top_slots,
)

SLOT_DIMS = 3 + N_TICKET_KINDS          # severity, age, kind one-hot, assigned
VENDOR_DIMS = 2 + N_TICKET_KINDS        # available, cost, kind one-hot
OBS_SIZE = K_SLOTS * SLOT_DIMS + 3 * VENDOR_DIMS + 3


class DispatchObs(ObsBuilder[int, np.ndarray, DispatchState, int]):
    """56 normalized dims. Slot order comes from top_slots — the same authority
    the engine uses to resolve actions, so what the agent sees is what it acts on.
    The week clock is encoded as (cos, sin): a raw tick count would present the
    Sunday->Monday wrap as a cliff, the same discontinuity as a raw yaw angle.
    """

    def get_obs_space(self, agent: int) -> int:
        return OBS_SIZE

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[int], state: DispatchState,
                  shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        offset = 0
        for slot_index in top_slots(state):
            ticket = state.tickets[slot_index]
            obs[offset + 0] = ticket.severity / MAX_SEVERITY
            obs[offset + 1] = min(ticket.age_hours / MAX_AGE_NORM, 1.0)
            obs[offset + 2 + ticket.kind] = 1.0
            obs[offset + 2 + N_TICKET_KINDS] = float(ticket.assigned_vendor is not None)
            offset += SLOT_DIMS
        offset = K_SLOTS * SLOT_DIMS
        for vendor in state.vendors:
            obs[offset + 0] = float(vendor.available)
            obs[offset + 1] = vendor.cost_per_job / MAX_COST_NORM
            obs[offset + 2 + vendor.kind] = 1.0
            offset += VENDOR_DIMS
        angle = 2.0 * np.pi * (state.ticks % WEEK_TICKS) / WEEK_TICKS
        obs[-3] = min(state.budget_spent / MAX_BUDGET_NORM, 1.0)
        obs[-2] = np.cos(angle)
        obs[-1] = np.sin(angle)
        return {agent: obs.copy() for agent in agents}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_obs.py -v`
Expected: 6 passed. Full suite: 134 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch/obs.py tests/test_dispatch_obs.py
git commit -m "feat: add 56-dim dispatch observation builder

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Action parser

**Files:**
- Create: `domains/dispatch/actions.py`
- Test: `tests/test_dispatch_actions.py`

**Interfaces:**
- Produces: `DispatchAction(ActionParser[int, int, int, DispatchState, int])` re-exporting `N_ACTIONS = 16` and `DEFER_ACTION = 15` from the engine module. Pass-through parser: policy int → engine int, accepting numpy scalars.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_actions.py
import numpy as np
from domains.dispatch.state import DispatchState, K_SLOTS, N_VENDORS
from domains.dispatch.actions import DispatchAction, N_ACTIONS, DEFER_ACTION


def blank():
    return DispatchState(tickets=[], vendors=[])


def test_action_space_is_sixteen():
    assert N_ACTIONS == K_SLOTS * N_VENDORS + 1 == 16
    assert DEFER_ACTION == 15
    assert DispatchAction().get_action_space(0) == 16


def test_parse_is_identity_and_accepts_numpy():
    parser = DispatchAction()
    parser.reset([0], blank(), {})
    assert parser.parse_actions({0: 7}, blank(), {}) == {0: 7}
    assert parser.parse_actions({0: np.int64(3)}, blank(), {}) == {0: 3}
    assert type(parser.parse_actions({0: np.int64(3)}, blank(), {})[0]) is int
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_actions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/actions.py
"""Action parser. Actions are already engine-shaped ints; this just casts and
documents the space: a = slot * N_VENDORS + vendor, or DEFER_ACTION."""
from typing import Any, Dict, List

from rlgym.api import ActionParser

from domains.dispatch.engine import DEFER_ACTION, N_ACTIONS
from domains.dispatch.state import DispatchState

__all__ = ["DispatchAction", "N_ACTIONS", "DEFER_ACTION"]


class DispatchAction(ActionParser[int, int, int, DispatchState, int]):
    def get_action_space(self, agent: int) -> int:
        return N_ACTIONS

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[int, int], state: DispatchState,
                      shared_info: Dict[str, Any]) -> Dict[int, int]:
        return {agent: int(action) for agent, action in actions.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_actions.py -v`
Expected: 2 passed. Full suite: 136 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch/actions.py tests/test_dispatch_actions.py
git commit -m "feat: add dispatch action parser over the 16-action space

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Reward functions

**Files:**
- Create: `domains/dispatch/rewards.py`
- Test: `tests/test_dispatch_rewards.py`

**Interfaces:**
- Produces: `OpenTicketBurden`, `CostPenalty`, `SlaResolutionReward`, `CatastropheReward`, and a local `CombinedReward` (deliberate per-domain copy — domains are self-contained costumes, mirroring how `rlgym.rocket_league` owns its own). All use the 5-arg `get_rewards`.

**The ladder (wired in Task 8):** burden 0.01 (dense), cost 0.05 (event), SLA 0.5 (event), catastrophe 10.0 (terminal). Components emit natural ranges; only `CombinedReward` applies weights.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_rewards.py
import pytest
from domains.dispatch.state import DispatchState, Ticket, MAX_SEVERITY
from domains.dispatch.rewards import (
    CatastropheReward, CombinedReward, CostPenalty, OpenTicketBurden,
    SlaResolutionReward,
)

NOT_DONE = ({0: False}, {0: False})


def make_state(tickets=(), resolved=(), spent=0.0, catastrophe=None):
    state = DispatchState(tickets=list(tickets), vendors=[])
    state.resolved_this_tick = list(resolved)
    state.spent_this_tick = spent
    state.catastrophe = catastrophe
    return state


def tk(sev):
    return Ticket(ticket_id=0, kind=0, severity=sev)


def test_open_ticket_burden_scales_with_severity():
    reward = OpenTicketBurden()
    reward.reset([0], make_state(), {})
    empty = reward.get_rewards([0], make_state(), *NOT_DONE, {})[0]
    heavy = reward.get_rewards([0], make_state([tk(5), tk(5)]), *NOT_DONE, {})[0]
    assert empty == 0.0
    assert heavy == pytest.approx(-2.0)            # -(5/5) per severity-5 ticket


def test_cost_penalty_is_negative_per_dollar():
    reward = CostPenalty()
    reward.reset([0], make_state(), {})
    assert reward.get_rewards([0], make_state(spent=200.0), *NOT_DONE, {})[0] \
        == pytest.approx(-2.0)                     # -spent/100


def test_sla_reward_pays_full_in_sla_and_less_late():
    reward = SlaResolutionReward()
    reward.reset([0], make_state(), {})
    both = make_state(resolved=[(3, True), (5, False)])
    assert reward.get_rewards([0], both, *NOT_DONE, {})[0] == pytest.approx(1.3)


def test_catastrophe_reward_fires_negative_once():
    reward = CatastropheReward()
    reward.reset([0], make_state(), {})
    assert reward.get_rewards([0], make_state(), *NOT_DONE, {})[0] == 0.0
    assert reward.get_rewards([0], make_state(catastrophe=2),
                              {0: True}, {0: False}, {})[0] == -1.0


def test_combined_applies_weights_discriminably():
    # Components must evaluate to DIFFERENT values or swapped weights are
    # invisible (the exact test bug the MiniRL review caught).
    state = make_state(spent=40.0, catastrophe=0)   # cost=-0.4, catastrophe=-1
    correct = CombinedReward((CatastropheReward(), 10.0), (CostPenalty(), 0.05))
    correct.reset([0], make_state(), {})
    swapped = CombinedReward((CatastropheReward(), 0.05), (CostPenalty(), 10.0))
    swapped.reset([0], make_state(), {})
    assert correct.get_rewards([0], state, {0: True}, {0: False}, {})[0] \
        == pytest.approx(-10.02)
    assert swapped.get_rewards([0], state, {0: True}, {0: False}, {})[0] \
        == pytest.approx(-4.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_rewards.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/rewards.py
"""Reward functions. Same ladder shape as MiniRL: dense terms are training
wheels and must never outweigh the terminal event.

Built-in hacking demo for Stage-3-style experiments: replace
SlaResolutionReward with raw closure count and watch the agent spam cheap
vendors at trivial tickets while emergencies burn.
"""
from typing import Any, Dict, List, Tuple

from rlgym.api import RewardFunction

from domains.dispatch.state import MAX_SEVERITY, DispatchState


class OpenTicketBurden(RewardFunction[int, DispatchState, float]):
    """Dense: -(severity/MAX_SEVERITY) per open ticket, every hour."""

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: DispatchState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        burden = -sum(t.severity for t in state.tickets) / MAX_SEVERITY
        return {agent: burden for agent in agents}


class CostPenalty(RewardFunction[int, DispatchState, float]):
    """Event: -(dollars spent this tick)/100."""

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: DispatchState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        return {agent: -state.spent_this_tick / 100.0 for agent in agents}


class SlaResolutionReward(RewardFunction[int, DispatchState, float]):
    """Event: +1 per in-SLA fix, +0.3 per late fix, this tick."""

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: DispatchState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        total = sum(1.0 if in_sla else 0.3 for _, in_sla in state.resolved_this_tick)
        return {agent: total for agent in agents}


class CatastropheReward(RewardFunction[int, DispatchState, float]):
    """Terminal: -1 when the building blows up. The episode ends with it."""

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: DispatchState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        value = -1.0 if state.catastrophe is not None else 0.0
        return {agent: value for agent in agents}


class CombinedReward(RewardFunction[int, DispatchState, float]):
    """Weighted sum. Deliberate per-domain copy: domains are self-contained."""

    def __init__(self, *weighted_rewards: Tuple[RewardFunction, float]) -> None:
        self.weighted_rewards = weighted_rewards

    def reset(self, agents: List[int], initial_state: DispatchState,
              shared_info: Dict[str, Any]) -> None:
        for reward_fn, _ in self.weighted_rewards:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[int], state: DispatchState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        totals = {agent: 0.0 for agent in agents}
        for reward_fn, weight in self.weighted_rewards:
            component = reward_fn.get_rewards(agents, state, is_terminated,
                                              is_truncated, shared_info)
            for agent in agents:
                totals[agent] += weight * component[agent]
        return totals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_rewards.py -v`
Expected: 5 passed. Full suite: 141 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch/rewards.py tests/test_dispatch_rewards.py
git commit -m "feat: add dispatch reward ladder

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: build_env, episode events, and integration

**Files:**
- Create: `domains/dispatch/env.py`
- Test: `tests/test_dispatch_env.py`

**Interfaces:**
- Produces: `build_env(rng=None) -> RLGym`, `episode_events(state) -> {"catastrophe": bool, "sla_clean": bool}`, re-exported `OBS_SIZE`, `N_ACTIONS`, `EVENT_NAMES = ("catastrophe", "sla_clean")`. `sla_clean` = at least one fix and zero late fixes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_env.py
import numpy as np
from domains.dispatch.env import (
    EVENT_NAMES, N_ACTIONS, OBS_SIZE, build_env, episode_events,
)
from domains.dispatch.state import WEEK_TICKS


def test_reset_and_spaces():
    env = build_env(rng=np.random.default_rng(0))
    obs = env.reset()
    assert set(obs.keys()) == {0} and obs[0].shape == (OBS_SIZE,)
    assert env.action_space(0) == N_ACTIONS
    assert env.observation_space(0) == OBS_SIZE


def test_step_returns_four_tuple():
    env = build_env(rng=np.random.default_rng(0))
    env.reset()
    result = env.step({0: 0})
    assert len(result) == 4
    obs, rewards, terminated, truncated = result
    assert isinstance(rewards[0], float)


def test_random_week_is_stable_and_finite():
    rng = np.random.default_rng(1)
    env = build_env(rng=rng)
    env.reset()
    steps = 0
    for _ in range(WEEK_TICKS * 3):
        obs, rewards, terminated, truncated = env.step({0: int(rng.integers(N_ACTIONS))})
        assert np.all(np.isfinite(obs[0])) and np.isfinite(rewards[0])
        steps += 1
        if terminated[0] or truncated[0]:
            break
    assert steps <= WEEK_TICKS


def test_deferring_all_week_eventually_ends_badly_or_times_out():
    env = build_env(rng=np.random.default_rng(2))
    env.reset()
    from domains.dispatch.actions import DEFER_ACTION
    for _ in range(WEEK_TICKS):
        obs, rewards, terminated, truncated = env.step({0: DEFER_ACTION})
        if terminated[0]:
            assert env.state.catastrophe is not None
            return
        if truncated[0]:
            return
    raise AssertionError("episode never ended")


def test_episode_events_names_and_semantics():
    env = build_env(rng=np.random.default_rng(0))
    env.reset()
    events = episode_events(env.state)
    assert set(events) == set(EVENT_NAMES)
    assert events["catastrophe"] is False
    env.state.catastrophe = 1
    assert episode_events(env.state)["catastrophe"] is True
    env.state.catastrophe = None
    env.state.resolved_in_sla, env.state.resolved_late = 3, 0
    assert episode_events(env.state)["sla_clean"] is True
    env.state.resolved_late = 1
    assert episode_events(env.state)["sla_clean"] is False


def test_seeded_envs_reproduce():
    def run(seed):
        env = build_env(rng=np.random.default_rng(seed))
        env.reset()
        rewards = []
        for step in range(50):
            _, r, term, trunc = env.step({0: step % N_ACTIONS})
            rewards.append(r[0])
            if term[0] or trunc[0]:
                break
        return rewards
    assert run(5) == run(5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_env.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/env.py
"""Assembles the six configuration objects into the DispatchSim environment."""
from typing import Optional

import numpy as np
from rlgym.api import RLGym

from domains.dispatch.actions import DEFER_ACTION, DispatchAction, N_ACTIONS
from domains.dispatch.dones import CatastropheCondition, WeekEndCondition
from domains.dispatch.engine import DispatchEngine
from domains.dispatch.mutators import WeekStartMutator
from domains.dispatch.obs import OBS_SIZE, DispatchObs
from domains.dispatch.rewards import (
    CatastropheReward, CombinedReward, CostPenalty, OpenTicketBurden,
    SlaResolutionReward,
)
from domains.dispatch.state import DispatchState

__all__ = ["build_env", "episode_events", "OBS_SIZE", "N_ACTIONS", "EVENT_NAMES"]

EVENT_NAMES = ("catastrophe", "sla_clean")


def episode_events(state: DispatchState) -> dict:
    """Episode-level events for core.rollout.VecEnv(event_fn=...)."""
    return {
        "catastrophe": state.catastrophe is not None,
        "sla_clean": state.resolved_late == 0 and state.resolved_in_sla > 0,
    }


def build_env(rng: Optional[np.random.Generator] = None) -> RLGym:
    rng = rng if rng is not None else np.random.default_rng()
    return RLGym(
        state_mutator=WeekStartMutator(rng=rng),
        obs_builder=DispatchObs(),
        action_parser=DispatchAction(),
        reward_fn=CombinedReward(
            (OpenTicketBurden(), 0.01),
            (CostPenalty(), 0.05),
            (SlaResolutionReward(), 0.5),
            (CatastropheReward(), 10.0),
        ),
        transition_engine=DispatchEngine(rng=rng),
        termination_cond=CatastropheCondition(),
        truncation_cond=WeekEndCondition(),
    )
```

Note the single `rng` is shared by mutator and engine deliberately — one seed,
one week. Reproducibility comes from construction order, which is fixed.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_env.py -v`
Expected: 6 passed. Full suite: 147 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch/env.py tests/test_dispatch_env.py
git commit -m "feat: assemble DispatchSim environment with episode events

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Board renderer

**Files:**
- Create: `domains/dispatch/render.py`
- Test: `tests/test_dispatch_render.py`

**Interfaces:**
- Produces: `draw_board(ax, state) -> None` — plugs into `core.dashboard.Dashboard(draw_fn=...)` unchanged. Ticket queue as horizontal severity bars (top_slots order, height-stacked), red on catastrophe; vendor status dots; budget text.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatch_render.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.dashboard import Dashboard
from domains.dispatch.render import draw_board
from domains.dispatch.state import DispatchState, Ticket, Vendor


def make_state(n_tickets=4, catastrophe=None):
    tickets = [Ticket(ticket_id=i, kind=i % 4, severity=(i % 5) + 1, age_hours=i * 3)
               for i in range(n_tickets)]
    vendors = [Vendor(kind=k, cost_per_job=100.0, no_show_prob=0.1) for k in range(3)]
    state = DispatchState(tickets=tickets, vendors=vendors)
    state.catastrophe = catastrophe
    return state


def test_draw_board_populates_axes_and_clears_between_frames():
    figure, ax = plt.subplots()
    draw_board(ax, make_state())
    first = len(ax.patches) + len(ax.lines) + len(ax.texts)
    assert first > 0
    draw_board(ax, make_state())
    assert len(ax.patches) + len(ax.lines) + len(ax.texts) == first
    plt.close(figure)


def test_draw_board_handles_empty_queue():
    figure, ax = plt.subplots()
    draw_board(ax, DispatchState(tickets=[], vendors=[]))
    plt.close(figure)


def test_dashboard_accepts_draw_board():
    dashboard = Dashboard(draw_fn=draw_board, title="DispatchSim")
    dashboard.show_episode([make_state(), make_state(catastrophe=2)])
    dashboard.update_curve(1, {"mean_episode_reward": 1.0, "episodes": 5,
                               "catastrophe_rate": 0.2, "sla_clean_rate": 0.4})
    assert dashboard.rates["catastrophe"] == [0.2]
    dashboard.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_dispatch_render.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/dispatch/render.py
"""Ticket-board drawing. Knows about tickets and vendors; nothing about training."""
from matplotlib.patches import Rectangle

from domains.dispatch.state import (
    K_SLOTS, KIND_NAMES, MAX_SEVERITY, DispatchState, top_slots,
)

KIND_COLOURS = ("tab:blue", "tab:olive", "tab:cyan", "tab:purple")


def draw_board(ax, state: DispatchState) -> None:
    """Draw one frame of the dispatch board onto `ax`. Clears the axes first."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, K_SLOTS + 2)
    ax.set_xticks([])
    ax.set_yticks([])

    burst = state.catastrophe is not None
    if burst:
        ax.set_facecolor("#ffe5e5")
        ax.text(0.5, K_SLOTS + 1.5, f"CATASTROPHE: {KIND_NAMES[state.catastrophe]}",
                ha="center", color="tab:red", fontsize=12, fontweight="bold")
    else:
        ax.set_facecolor("white")

    slots = top_slots(state)
    for row, slot_index in enumerate(slots):
        ticket = state.tickets[slot_index]
        y = K_SLOTS - row
        width = ticket.severity / MAX_SEVERITY
        # Older tickets darken toward red-ish urgency.
        age_shade = min(ticket.age_hours / 72.0, 1.0)
        colour = KIND_COLOURS[ticket.kind]
        ax.add_patch(Rectangle((0.0, y - 0.4), width, 0.8,
                               facecolor=colour, alpha=0.4 + 0.6 * age_shade,
                               edgecolor="0.3", linewidth=1.0))
        marker = "*" if ticket.assigned_vendor is not None else ""
        ax.text(width + 0.02, y, f"{KIND_NAMES[ticket.kind]} s{ticket.severity} "
                                 f"{ticket.age_hours}h{marker}",
                va="center", fontsize=8)

    for i, vendor in enumerate(state.vendors):
        ax.plot([0.05 + i * 0.12], [0.4], marker="o", markersize=10,
                color=("tab:green" if vendor.available else "tab:red"))
    ax.text(0.5, 0.4, f"${state.budget_spent:,.0f} | h{state.ticks}",
            fontsize=9, va="center")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_dispatch_render.py -v`
Expected: 3 passed. Full suite: 150 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/dispatch/render.py tests/test_dispatch_render.py
git commit -m "feat: add dispatch ticket-board renderer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: train.py --domain flag and checkpointing

**Files:**
- Modify: `train.py`
- Test: `tests/test_train_smoke.py` (extend)

**Interfaces:**
- Produces: `run_training(..., domain: str = "minirl", save_path: Optional[str] = None)`; a `DOMAINS` registry mapping name → `(build_env, draw_fn, event_fn, event_names, obs_size, n_actions)`; CLI `--domain {minirl,dispatch}` and `--save PATH`. On completion, saves `torch.save({"state_dict": ..., "obs_size": ..., "n_actions": ...}, save_path)` when `save_path` is set. `_record_episode` must use the domain's `build_env` (parameterize it — it currently hardcodes MiniRL's).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_smoke.py`:

```python
def test_training_runs_on_the_dispatch_domain(tmp_path):
    save = str(tmp_path / "dispatch.pt")
    history = run_training(iterations=2, n_envs=2, horizon=64, render_every=0,
                           show_dashboard=False, seed=0, domain="dispatch",
                           save_path=save)
    assert len(history) == 2
    for snapshot in history:
        assert "catastrophe_rate" in snapshot and "sla_clean_rate" in snapshot

    import torch
    payload = torch.load(save, weights_only=True)
    assert payload["obs_size"] == 56 and payload["n_actions"] == 16


def test_minirl_stays_the_default():
    history = run_training(iterations=1, n_envs=2, horizon=32, render_every=0,
                           show_dashboard=False, seed=0)
    assert "touch_rate" in history[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_train_smoke.py -v`
Expected: FAIL with `TypeError` (unexpected keyword `domain`)

- [ ] **Step 3: Write minimal implementation**

Rework `train.py`'s wiring (keep GAE/update loop identical):

```python
# train.py — registry + signature changes
from domains.minirl import env as minirl_env
from domains.minirl.render import draw_arena
from domains.dispatch import env as dispatch_env
from domains.dispatch.render import draw_board

DOMAINS = {
    "minirl": dict(build_env=minirl_env.build_env, draw_fn=draw_arena,
                   event_fn=minirl_env.episode_events, event_names=("touch", "goal"),
                   obs_size=minirl_env.OBS_SIZE, n_actions=minirl_env.N_ACTIONS),
    "dispatch": dict(build_env=dispatch_env.build_env, draw_fn=draw_board,
                     event_fn=dispatch_env.episode_events,
                     event_names=dispatch_env.EVENT_NAMES,
                     obs_size=dispatch_env.OBS_SIZE, n_actions=dispatch_env.N_ACTIONS),
}
```

`run_training(iterations, n_envs, horizon, render_every, show_dashboard, seed,
domain="minirl", save_path=None)` looks everything up from `DOMAINS[domain]`,
builds `EpisodeTracker(spec["event_names"])`, passes `event_fn` into `VecEnv`,
sizes `ActorCritic(spec["obs_size"], spec["n_actions"])`, flattens with
`spec["obs_size"]`, and prints a generic line built from the snapshot:

```python
        rates = " | ".join(f"{k[:-5]} {snapshot[k]:5.2f}"
                           for k in sorted(snapshot) if k.endswith("_rate"))
        print(f"iter {iteration:4d} | reward {snapshot['mean_episode_reward']:8.3f} "
              f"| {rates} | entropy {stats['entropy']:5.3f} "
              f"| episodes {snapshot['episodes']}")
```

**Regression-gate warning:** with names ("touch", "goal") sorted alphabetically,
the generic line prints `goal` BEFORE `touch`, which changes the console format
and breaks the byte-exact gate. Keep MiniRL's event ordering by iterating
`spec["event_names"]` (insertion order), NOT `sorted(snapshot)`:

```python
        rates = " | ".join(f"{name} {snapshot[f'{name}_rate']:5.2f}"
                           for name in spec["event_names"])
```

With `event_names=("touch", "goal")` this reproduces the exact historical
format. `_record_episode(model, build_env_fn, max_steps)` takes the env factory
as a parameter. At the end:

```python
    if save_path is not None:
        torch.save({"state_dict": model.state_dict(),
                    "obs_size": spec["obs_size"],
                    "n_actions": spec["n_actions"]}, save_path)
```

CLI adds `--domain {minirl,dispatch}` (default minirl) and `--save PATH`
(default None).

- [ ] **Step 4: Run tests and re-run the regression gate**

Run: `./.venv/bin/pytest tests/ -v`
Expected: **152 passed, 3 skipped**

Run: `./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0 | tail -1`
Expected — byte-for-byte:
`iter   12 | reward    0.172 | touch  0.13 | goal  0.00 | entropy 3.223 | episodes 100`

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_smoke.py
git commit -m "feat: multi-domain training with --domain flag and checkpointing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: evaluate.py — the harness

**Files:**
- Create: `evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `random_agent(obs, rng) -> int` and `fifo_agent(obs) -> int` (decodes the obs layout: oldest unassigned occupied slot → first available vendor, else defer).
  - `evaluate(agent_fn, n_scenarios=200, seed=0) -> dict` with keys `n_scenarios`, `catastrophe_rate`, `sla_rate` (in-SLA fixes / all fixes), `mean_cost`, `mean_reward`, `worst` (list of `{"seed_index", "reward", "catastrophe"}` for the 5 worst scenarios). Scenario i uses `np.random.default_rng(np.random.SeedSequence(seed).spawn(n_scenarios)[i])` so every listed failure replays exactly.
  - `load_policy_agent(checkpoint_path) -> agent_fn` (argmax over `ActorCritic` logits).
  - CLI: `--agent {random,fifo,policy} [--checkpoint PATH] [--scenarios N] [--seed N]`, prints a report table.
- `agent_fn` signature is `(obs: np.ndarray) -> int`; `random_agent` is wrapped at the call site to bind its rng.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
from evaluate import evaluate, fifo_agent, random_agent
from domains.dispatch.actions import DEFER_ACTION, N_ACTIONS
from domains.dispatch.env import build_env
from domains.dispatch.obs import SLOT_DIMS
from domains.dispatch.state import K_SLOTS


def test_evaluate_returns_the_report_shape():
    rng = np.random.default_rng(0)
    report = evaluate(lambda obs: int(rng.integers(N_ACTIONS)), n_scenarios=5, seed=0)
    assert report["n_scenarios"] == 5
    for key in ("catastrophe_rate", "sla_rate", "mean_cost", "mean_reward", "worst"):
        assert key in report
    assert 0.0 <= report["catastrophe_rate"] <= 1.0
    assert len(report["worst"]) <= 5


def test_evaluate_is_reproducible():
    a = evaluate(lambda obs: DEFER_ACTION, n_scenarios=4, seed=3)
    b = evaluate(lambda obs: DEFER_ACTION, n_scenarios=4, seed=3)
    assert a == b


def test_fifo_agent_assigns_rather_than_defers_when_work_exists():
    env = build_env(rng=np.random.default_rng(0))
    obs = env.reset()[0]
    if np.any(obs[:K_SLOTS * SLOT_DIMS] != 0.0):     # backlog exists at week start
        assert fifo_agent(obs) != DEFER_ACTION


def test_fifo_agent_defers_on_an_empty_board():
    empty = np.zeros(56, dtype=np.float32)
    empty[35] = 1.0                                   # one available vendor
    assert fifo_agent(empty) == DEFER_ACTION


def test_fifo_beats_pure_deferral():
    defer_report = evaluate(lambda obs: DEFER_ACTION, n_scenarios=20, seed=0)
    fifo_report = evaluate(fifo_agent, n_scenarios=20, seed=0)
    assert fifo_report["catastrophe_rate"] < defer_report["catastrophe_rate"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluate.py
"""Run any (obs) -> action agent through seeded DispatchSim scenarios.

This is the demo-asset half of the project: the same six objects that train a
policy also stress-test a finished agent. Every scenario is seeded, so every
failure in the report replays exactly.
"""
import argparse
from typing import Callable, Dict, List

import numpy as np
import torch

from core.ppo import ActorCritic
from domains.dispatch.actions import DEFER_ACTION, N_ACTIONS
from domains.dispatch.env import build_env
from domains.dispatch.obs import SLOT_DIMS, VENDOR_DIMS
from domains.dispatch.state import K_SLOTS, N_VENDORS, WEEK_TICKS

AgentFn = Callable[[np.ndarray], int]


def random_agent(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(N_ACTIONS))


def fifo_agent(obs: np.ndarray) -> int:
    """Assign the oldest unassigned visible ticket to the first free vendor."""
    vendor_base = K_SLOTS * SLOT_DIMS
    free_vendors = [v for v in range(N_VENDORS)
                    if obs[vendor_base + v * VENDOR_DIMS] > 0.5]
    if not free_vendors:
        return DEFER_ACTION
    best_slot, best_age = None, -1.0
    for slot in range(K_SLOTS):
        base = slot * SLOT_DIMS
        occupied = obs[base] > 0.0                    # severity > 0
        assigned = obs[base + 2 + 4] > 0.5            # assigned flag
        if occupied and not assigned and obs[base + 1] > best_age:
            best_slot, best_age = slot, float(obs[base + 1])
    if best_slot is None:
        return DEFER_ACTION
    return best_slot * N_VENDORS + free_vendors[0]


def load_policy_agent(checkpoint_path: str) -> AgentFn:
    payload = torch.load(checkpoint_path, weights_only=True)
    model = ActorCritic(payload["obs_size"], payload["n_actions"])
    model.load_state_dict(payload["state_dict"])
    model.eval()

    def agent(obs: np.ndarray) -> int:
        with torch.no_grad():
            logits = model.policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())
    return agent


def evaluate(agent_fn: AgentFn, n_scenarios: int = 200, seed: int = 0) -> Dict:
    seeds = np.random.SeedSequence(seed).spawn(n_scenarios)
    results: List[Dict] = []
    for index in range(n_scenarios):
        env = build_env(rng=np.random.default_rng(seeds[index]))
        obs = env.reset()
        total_reward = 0.0
        for _ in range(WEEK_TICKS):
            action = agent_fn(obs[0])
            obs, rewards, terminated, truncated = env.step({0: action})
            total_reward += rewards[0]
            if terminated[0] or truncated[0]:
                break
        state = env.state
        fixes = state.resolved_in_sla + state.resolved_late
        results.append({
            "seed_index": index,
            "reward": round(total_reward, 4),
            "catastrophe": state.catastrophe is not None,
            "in_sla": state.resolved_in_sla,
            "fixes": fixes,
            "cost": state.budget_spent,
        })
    total_fixes = sum(r["fixes"] for r in results)
    report = {
        "n_scenarios": n_scenarios,
        "catastrophe_rate": sum(r["catastrophe"] for r in results) / n_scenarios,
        "sla_rate": (sum(r["in_sla"] for r in results) / total_fixes) if total_fixes else 0.0,
        "mean_cost": sum(r["cost"] for r in results) / n_scenarios,
        "mean_reward": round(sum(r["reward"] for r in results) / n_scenarios, 4),
        "worst": sorted(
            ({"seed_index": r["seed_index"], "reward": r["reward"],
              "catastrophe": r["catastrophe"]} for r in results),
            key=lambda r: r["reward"])[:5],
    }
    return report


def format_report(name: str, report: Dict) -> str:
    lines = [
        f"=== {name} over {report['n_scenarios']} scenarios ===",
        f"catastrophe rate : {report['catastrophe_rate']:.2%}",
        f"in-SLA fix rate  : {report['sla_rate']:.2%}",
        f"mean cost/week   : ${report['mean_cost']:,.0f}",
        f"mean reward      : {report['mean_reward']:.3f}",
        "worst scenarios  : " + ", ".join(
            f"#{w['seed_index']}({w['reward']:.1f}{'!' if w['catastrophe'] else ''})"
            for w in report["worst"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an agent in DispatchSim.")
    parser.add_argument("--agent", choices=("random", "fifo", "policy"), default="fifo")
    parser.add_argument("--checkpoint", default="checkpoints/dispatch.pt")
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.agent == "random":
        rng = np.random.default_rng(args.seed)
        agent_fn: AgentFn = lambda obs: random_agent(obs, rng)
    elif args.agent == "fifo":
        agent_fn = fifo_agent
    else:
        agent_fn = load_policy_agent(args.checkpoint)

    print(format_report(args.agent, evaluate(agent_fn, args.scenarios, args.seed)))
```

Fix the obvious transcription slip in `evaluate`: the step line is plainly
`obs, rewards, terminated, truncated = env.step({0: action})` — no `+ ()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_evaluate.py -v`
Expected: 5 passed. Full suite: 157 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: add seeded evaluation harness with fifo/random baselines

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Train it, beat the baselines, hold the regression

**Files:**
- Create: `checkpoints/` (gitignored — verify `.gitignore` already covers it)
- Test: none new; this task produces measured numbers.

**Interfaces:** consumes everything.

- [ ] **Step 1: Measure the baselines**

```bash
./.venv/bin/python evaluate.py --agent random --scenarios 200 --seed 0
./.venv/bin/python evaluate.py --agent fifo --scenarios 200 --seed 0
```

Record all numbers. Expected shape: fifo has a much lower catastrophe rate than
random; both nonzero.

- [ ] **Step 2: Train the dispatch policy**

```bash
./.venv/bin/python train.py --domain dispatch --iterations 300 --render-every 0 --no-dashboard --seed 0 --save checkpoints/dispatch.pt
```

Watch `catastrophe` fall and `reward` rise across iterations. Judge over the
full run, not the first 60 — the curve is non-monotonic early (MiniRL lesson).

- [ ] **Step 3: Evaluate the trained policy on the SAME scenario seeds**

```bash
./.venv/bin/python evaluate.py --agent policy --checkpoint checkpoints/dispatch.pt --scenarios 200 --seed 0
```

**Success criteria (from the spec):** policy beats `fifo` on catastrophe rate
and beats `random` on every metric. If the policy loses to fifo: ONE tuning
pass is budgeted — adjust reward weights only (not code), re-train, re-measure,
and document what changed and why in the commit message. If it still loses,
STOP and report the numbers honestly rather than tuning in circles.

- [ ] **Step 4: Hold the MiniRL regression one final time**

```bash
./.venv/bin/pytest tests/ -q
./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0 | tail -1
```

Expected: 157 passed, 3 skipped; and the byte-exact iter-12 line.

- [ ] **Step 5: Commit the results**

```bash
git add -A
git commit -m "docs: record dispatch training results and baseline comparison

<paste the three report tables into the commit body>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** core generalization + regression gate → Task 1. State/constants →
Task 2 (vendor "specialty = 2× faster" refinement carried from the amended spec).
Engine (arrivals, no-shows, escalation, SLA) → Task 3. Mutator + dones → Task 4.
56-dim obs → Task 5. 16-action space → Task 6. Reward ladder → Task 7 (with a
discriminating weights test — the MiniRL lesson pre-applied). `build_env` +
`event_fn` → Task 8. Renderer → Task 9. `--domain` + checkpoints → Task 10
(insertion-order rates preserve the byte-exact console format). Harness +
baselines → Task 11. Training vs baselines + final regression → Task 12.

**Known deliberate choices:** `CombinedReward` duplicated per-domain (domains are
self-contained; mirrors rlgym.rocket_league's own layout). One shared `rng`
between mutator and engine per env (one seed = one reproducible week).
`evaluate.py` decodes obs slices via named constants from `obs.py`
(`SLOT_DIMS`/`VENDOR_DIMS` must be exported there — Task 5 defines them at module
level, so they are importable).

**Type consistency:** `top_slots` defined once (Task 2), consumed by engine
(Task 3), obs (Task 5). `N_ACTIONS`/`DEFER_ACTION` defined in engine (Task 3),
re-exported by actions (Task 6) and env (Task 8), consumed by evaluate (Task 11).
`EpisodeTracker(event_names)` (Task 1) consumed by train.py (Task 10).
`resolved_this_tick: List[Tuple[int, bool]]` written by engine (Task 3), read by
`SlaResolutionReward` (Task 7). Checkpoint payload keys (Task 10) match
`load_policy_agent` (Task 11).
