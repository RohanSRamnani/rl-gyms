# ContentSim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `core/` to be genuinely domain-agnostic, then build a simulated-TikTok creator domain (`domains/content/`) plus an evaluation harness (`evaluate.py`) on the unchanged core.

**Architecture:** `core/` loses its MiniRL vocabulary via a domain-supplied `event_fn(state) -> Dict[str, bool]`. `domains/content/` implements the six RLGym objects for a one-account content simulator (1 tick = 1 day, 30-day episodes, hidden drifting "meta"). `train.py` gains a `--domain` flag; `evaluate.py` runs any `(obs) -> action` agent through seeded scenarios and reports failures.

**Tech Stack:** Python 3.13, `rlgym` 2.0.1, numpy, torch, matplotlib, pytest — all already in `./.venv`.

**Spec:** `docs/superpowers/specs/2026-08-09-content-sim-design.md`

## Global Constraints

- All commands run from `/Users/rohan/Projects/RL Gyms` using `./.venv/bin/python` / `./.venv/bin/pytest`. Never pip install.
- `core/` must not import from `domains/` (enforced by `tests/test_boundary.py`) **and must not reference domain attribute names** — that is the leak Task 1 removes.
- **Regression gate:** BEFORE starting Task 1, run
  `./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0 | tail -1`
  and record the line. Expected (verified when the DispatchSim plan was written):
  `iter   12 | reward    0.172 | touch  0.13 | goal  0.00 | entropy 3.223 | episodes 100`.
  If the pre-refactor output differs, the recorded pre-refactor line becomes the gate. After Task 1 (and again after Task 10 and Task 12) the same command must reproduce that line **byte-for-byte**. Any deviation means the refactor changed behaviour, not just structure — a bug, full stop.
- Content domain: 1 tick = 1 day, no action repeat; `CYCLE_TICKS = 30` truncates; restriction (3 flag strikes) terminates. Obs is 49 dims; action space is 37.
- `decode_action` in `domains/content/state.py` is THE single action authority — engine, action parser, and scripted baselines all decode through it. A divergence silently breaks the action space.
- Hidden state (`meta`, `topic_heat`, `hook_fit`) must NEVER appear in the observation — only the noisy `trend_signals` / `heat_signals` do.
- Class/file naming mirrors `domains/minirl/` file-for-file.
- Every task ends with a commit. Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

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

- [ ] **Step 0: Capture the pre-refactor gate line**

Run: `./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0 | tail -1`
Record the exact output. Every later gate check compares against this line byte-for-byte.

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
    ("outlier", "restricted")). Each event is OR-accumulated across an
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
    PPO needs to bootstrap truncated episodes correctly.
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
Expected — byte-for-byte the Step 0 line:
`iter   12 | reward    0.172 | touch  0.13 | goal  0.00 | entropy 3.223 | episodes 100`

If it differs in ANY digit, the refactor changed behaviour. Debug; do not proceed, do not re-tune.

- [ ] **Step 10: Commit**

```bash
git add core/ domains/minirl/env.py train.py tests/
git commit -m "refactor: core/ event plumbing is domain-supplied, not hardcoded

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Content state, constants, and the shared action decoding

**Files:**
- Create: `domains/content/__init__.py` (empty), `domains/content/state.py`
- Test: `tests/test_content_state.py`

**Interfaces:**
- Produces: dataclasses `PostRecord`, `ContentState`; helper `decode_action(action) -> Optional[Tuple[int, int, int]]`; constants below. `decode_action` is THE action authority — engine, parser, and baselines all use it. `N_ACTIONS = 37`, `SKIP_ACTION = 36`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_state.py
import numpy as np
from domains.content.state import (
    PostRecord, ContentState, decode_action,
    N_FORMATS, N_HOOKS, N_TOPICS, N_ACTIONS, SKIP_ACTION, HISTORY_K, CYCLE_TICKS,
    V0, F0, REACH_EXPONENT, VIEW_NOISE_SIGMA, FOLLOW_RATE,
    INIT_FOLLOWERS_MIN, INIT_FOLLOWERS_MAX,
    AFFINITY_MIN, AFFINITY_MAX, AFFINITY_INIT_MIN, AFFINITY_INIT_MAX,
    DRIFT_STD, JUMP_PROB, SIGNAL_NOISE_STD,
    FATIGUE_DECAY, MAX_FATIGUE_NORM,
    AUTHORITY_GAIN, AUTHORITY_DECAY, AUTHORITY_SKIP_DECAY, AUTHORITY_MAX, AUTHORITY_BOOST,
    REPEAT_FLAG_STREAK, STRIKE_PROB, MAX_STRIKES,
    OUTLIER_MULTIPLIER, MIN_PRIOR_POSTS,
    MAX_FOLLOWERS_NORM, VIEWS_REF, FOLLOWS_REF,
)


def test_constants_are_coherent():
    assert N_ACTIONS == N_FORMATS * N_HOOKS * N_TOPICS + 1 == 37
    assert SKIP_ACTION == N_ACTIONS - 1
    assert CYCLE_TICKS == 30 and HISTORY_K == 3
    assert AFFINITY_MIN < AFFINITY_INIT_MIN < AFFINITY_INIT_MAX < AFFINITY_MAX
    assert 0 < STRIKE_PROB < 1 and 0 < JUMP_PROB < 1
    assert 0 < FATIGUE_DECAY < 1
    assert 0 < AUTHORITY_SKIP_DECAY < AUTHORITY_DECAY < 1
    assert INIT_FOLLOWERS_MIN < INIT_FOLLOWERS_MAX < MAX_FOLLOWERS_NORM
    assert MIN_PRIOR_POSTS >= 1 and OUTLIER_MULTIPLIER > 1.0
    assert REPEAT_FLAG_STREAK >= 2 and MAX_STRIKES >= 1


def test_decode_action_roundtrips_every_post_action():
    seen = set()
    for action in range(N_ACTIONS - 1):
        decoded = decode_action(action)
        assert decoded is not None
        f, h, t = decoded
        assert 0 <= f < N_FORMATS and 0 <= h < N_HOOKS and 0 <= t < N_TOPICS
        seen.add(decoded)
    assert len(seen) == N_FORMATS * N_HOOKS * N_TOPICS


def test_decode_action_skip_is_none():
    assert decode_action(SKIP_ACTION) is None


def test_state_constructs_with_defaults():
    state = ContentState()
    assert state.ticks == 0 and state.flag_strikes == 0
    assert state.restricted is False and state.had_outlier is False
    assert state.history == [] and state.posted_today is False
    assert state.views_today == 0.0 and state.follows_today == 0.0
    assert state.meta.shape == (N_FORMATS,)
    assert state.topic_heat.shape == (N_TOPICS,)
    assert state.hook_fit.shape == (N_FORMATS, N_HOOKS)
    assert state.trend_signals.shape == (N_FORMATS,)
    assert state.heat_signals.shape == (N_TOPICS,)
    assert state.fatigue.shape == (N_FORMATS,)
    assert state.authority.shape == (N_TOPICS,)
    assert state.last_action is None and state.repeat_streak == 0


def test_post_record_fields():
    record = PostRecord(format=1, hook=2, topic=0, day=5, views=1234.0, follows=12.0)
    assert (record.format, record.hook, record.topic) == (1, 2, 0)
    assert record.day == 5 and record.views == 1234.0 and record.follows == 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'domains.content'`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/state.py
"""State types and tuning constants for ContentSim.

1 tick = 1 simulated day. One episode = one 30-day strategy cycle.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# --- Scale -------------------------------------------------------------------
N_FORMATS = 3
FORMAT_NAMES = ("slideshow", "video", "talking_head")
N_HOOKS = 4
HOOK_NAMES = ("curiosity_list", "pov", "question", "transformation")
N_TOPICS = 3
TOPIC_NAMES = ("home_styling", "slow_living", "product_story")
HISTORY_K = 3                       # posts visible to the agent
CYCLE_TICKS = 30                    # one episode = one strategy cycle

# --- Action space ------------------------------------------------------------
N_ACTIONS = N_FORMATS * N_HOOKS * N_TOPICS + 1   # 36 posts + skip = 37
SKIP_ACTION = N_ACTIONS - 1

# --- Audience model ----------------------------------------------------------
V0 = 2000.0                         # base views at F0 followers, quality 1
F0 = 5000.0                         # reference follower count
REACH_EXPONENT = 0.6                # sublinear reach growth
VIEW_NOISE_SIGMA = 0.9              # lognormal sigma — the heavy tail is load-bearing
FOLLOW_RATE = 0.01                  # follows = rate * views * min(quality, 2)
INIT_FOLLOWERS_MIN = 1_000.0
INIT_FOLLOWERS_MAX = 20_000.0

# --- Hidden dynamics ---------------------------------------------------------
AFFINITY_MIN, AFFINITY_MAX = 0.2, 2.0            # clip range for meta / topic_heat
AFFINITY_INIT_MIN, AFFINITY_INIT_MAX = 0.5, 1.5  # per-episode init range
DRIFT_STD = 0.04                    # per-day random-walk step
JUMP_PROB = 0.03                    # per-day chance a format's meta reshuffles
SIGNAL_NOISE_STD = 0.15             # obs noise on trend/heat signals

# --- Fatigue / authority (the convergence tension) ---------------------------
FATIGUE_DECAY = 0.7                 # per-day multiplier; +1 per post of a format
MAX_FATIGUE_NORM = 4.0              # obs normalizer (steady-state ~2.3 for daily)
AUTHORITY_GAIN = 0.15               # per post of a topic
AUTHORITY_DECAY = 0.9               # per day when a post was made
AUTHORITY_SKIP_DECAY = 0.8          # per day when the account skipped
AUTHORITY_MAX = 1.0
AUTHORITY_BOOST = 0.5               # views multiplier = 1 + boost * authority

# --- Flag mechanic (unoriginal-content analog) -------------------------------
REPEAT_FLAG_STREAK = 3              # identical posts in a row before rolls start
STRIKE_PROB = 0.25                  # per qualifying day
MAX_STRIKES = 3                     # strikes -> restricted -> termination

# --- Outliers ----------------------------------------------------------------
OUTLIER_MULTIPLIER = 2.0            # views >= 2x trailing median
MIN_PRIOR_POSTS = 3                 # median needs a base

# --- Normalizers / reward references -----------------------------------------
MAX_FOLLOWERS_NORM = 200_000.0
VIEWS_REF = 50_000.0
FOLLOWS_REF = 500.0


@dataclass
class PostRecord:
    format: int
    hook: int
    topic: int
    day: int
    views: float
    follows: float


@dataclass
class ContentState:
    followers: float = 5000.0
    # HIDDEN — the environment's truth; must never leak into the obs.
    meta: np.ndarray = field(default_factory=lambda: np.ones(N_FORMATS))
    topic_heat: np.ndarray = field(default_factory=lambda: np.ones(N_TOPICS))
    hook_fit: np.ndarray = field(default_factory=lambda: np.ones((N_FORMATS, N_HOOKS)))
    # OBSERVED — noisy daily readings of the hidden state (the "research corpus").
    trend_signals: np.ndarray = field(default_factory=lambda: np.ones(N_FORMATS))
    heat_signals: np.ndarray = field(default_factory=lambda: np.ones(N_TOPICS))
    # Account condition.
    fatigue: np.ndarray = field(default_factory=lambda: np.zeros(N_FORMATS))
    authority: np.ndarray = field(default_factory=lambda: np.zeros(N_TOPICS))
    flag_strikes: int = 0
    restricted: bool = False
    had_outlier: bool = False
    history: List[PostRecord] = field(default_factory=list)
    ticks: int = 0
    # Streak bookkeeping for the flag mechanic.
    last_action: Optional[Tuple[int, int, int]] = None
    repeat_streak: int = 0
    # Per-tick reward channels, cleared at the start of every engine tick.
    views_today: float = 0.0
    follows_today: float = 0.0
    outlier_today: bool = False
    posted_today: bool = False


def decode_action(action: int) -> Optional[Tuple[int, int, int]]:
    """Action int -> (format, hook, topic), or None for skip.

    THE single action authority: the engine resolves actions through this and
    the scripted baselines encode with its inverse. If they ever diverge, the
    agent posts a different package than the one it chose.
    """
    if action == SKIP_ACTION:
        return None
    format_index, rest = divmod(action, N_HOOKS * N_TOPICS)
    hook_index, topic_index = divmod(rest, N_TOPICS)
    return format_index, hook_index, topic_index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_content_state.py -v`
Expected: 5 passed. Full suite: 112 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content tests/test_content_state.py
git commit -m "feat: add ContentSim state, constants, and action decoding

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: ContentEngine

**Files:**
- Create: `domains/content/engine.py`
- Test: `tests/test_content_engine.py`

**Interfaces:**
- Consumes: everything from Task 2.
- Produces: `ContentEngine(TransitionEngine[int, ContentState, int])`, constructed with `rng: Optional[np.random.Generator]`. Engine action is a plain `int` in `[0, 37)`, resolved via `decode_action`.

**Tick order inside `step` (one call = one day):** clear per-tick channels → apply action (streak bookkeeping; if a post: compute views/follows, outlier check, append history, update followers/fatigue/authority) → flag roll → hidden drift (+ possible jump) → daily decay (fatigue always; authority by posted/skip rate) → refresh noisy signals → `ticks += 1`.

**The views formula (the whole audience model):**

```
quality = meta[f] * topic_heat[t] * hook_fit[f, h]
          * (1 + AUTHORITY_BOOST * authority[t]) / (1 + fatigue[f])
views   = V0 * (followers / F0) ** REACH_EXPONENT * quality * exp(N(0, VIEW_NOISE_SIGMA))
follows = FOLLOW_RATE * views * min(quality, 2.0)
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_engine.py
import numpy as np
from domains.content.state import (
    ContentState, PostRecord, decode_action,
    N_ACTIONS, SKIP_ACTION, MAX_STRIKES, MIN_PRIOR_POSTS, CYCLE_TICKS,
    AFFINITY_MIN, AFFINITY_MAX,
)
from domains.content.engine import ContentEngine


def seeded_engine(seed=0, **overrides):
    engine = ContentEngine(rng=np.random.default_rng(seed))
    state = engine.create_base_state()
    for key, value in overrides.items():
        setattr(state, key, value)
    engine.set_state(state, {})
    return engine


POST_A = 0                       # (format 0, hook 0, topic 0)
POST_B = 1 * 12 + 2 * 3 + 1      # (format 1, hook 2, topic 1) = action 19


def test_agents_and_max():
    engine = seeded_engine()
    assert engine.agents == [0] and engine.max_num_agents == 1


def test_posting_records_views_and_grows_followers():
    engine = seeded_engine()
    before = engine.state.followers
    engine.step({0: POST_A}, {})
    state = engine.state
    assert state.posted_today and state.views_today > 0.0
    assert len(state.history) == 1
    assert state.history[0].views == state.views_today
    assert state.followers > before
    assert state.fatigue[0] > 0.0 and state.authority[0] > 0.0
    assert state.ticks == 1


def test_skip_posts_nothing_and_decays_authority_harder():
    posted = seeded_engine()
    posted.state.authority[:] = 0.5
    posted.step({0: POST_A}, {})
    skipped = seeded_engine()
    skipped.state.authority[:] = 0.5
    skipped.step({0: SKIP_ACTION}, {})
    assert skipped.state.views_today == 0.0 and skipped.state.history == []
    assert not skipped.state.posted_today
    # topic 2 was posted by neither engine; skip-day decay must bite harder.
    assert skipped.state.authority[2] < posted.state.authority[2]


def test_fatigue_lowers_views_all_else_equal():
    fresh = seeded_engine(seed=3)
    fresh.step({0: POST_A}, {})
    tired = seeded_engine(seed=3)
    tired.state.fatigue[0] = 3.0
    tired.step({0: POST_A}, {})
    assert tired.state.views_today < fresh.state.views_today


def test_authority_raises_views_all_else_equal():
    plain = seeded_engine(seed=4)
    plain.step({0: POST_A}, {})
    authoritative = seeded_engine(seed=4)
    authoritative.state.authority[0] = 1.0
    authoritative.step({0: POST_A}, {})
    assert authoritative.state.views_today > plain.state.views_today


def test_identical_streak_eventually_restricts():
    engine = seeded_engine(seed=0)
    for _ in range(60):
        engine.step({0: POST_A}, {})
        if engine.state.restricted:
            break
    assert engine.state.restricted and engine.state.flag_strikes >= MAX_STRIKES


def test_varied_posting_never_strikes():
    engine = seeded_engine(seed=0)
    for day in range(60):
        engine.step({0: POST_A if day % 2 == 0 else POST_B}, {})
    assert engine.state.flag_strikes == 0 and not engine.state.restricted


def test_outlier_fires_against_trailing_median():
    engine = seeded_engine(seed=1)
    engine.state.history = [
        PostRecord(format=0, hook=0, topic=0, day=d, views=1.0, follows=0.0)
        for d in range(MIN_PRIOR_POSTS)
    ]
    engine.step({0: POST_A}, {})
    # Median of priors is 1.0; any real post clears 2x that by orders of magnitude.
    assert engine.state.outlier_today and engine.state.had_outlier


def test_no_outlier_without_enough_priors():
    engine = seeded_engine(seed=1)
    engine.step({0: POST_A}, {})
    assert not engine.state.outlier_today


def test_hidden_state_drifts_and_stays_clipped():
    engine = seeded_engine(seed=2)
    start_meta = engine.state.meta.copy()
    for _ in range(CYCLE_TICKS):
        engine.step({0: SKIP_ACTION}, {})
    assert not np.allclose(engine.state.meta, start_meta)
    assert np.all(engine.state.meta >= AFFINITY_MIN)
    assert np.all(engine.state.meta <= AFFINITY_MAX)
    assert np.all(engine.state.topic_heat >= AFFINITY_MIN)
    assert np.all(engine.state.topic_heat <= AFFINITY_MAX)


def test_signals_track_hidden_state_noisily():
    engine = seeded_engine(seed=5)
    engine.step({0: SKIP_ACTION}, {})
    state = engine.state
    assert not np.array_equal(state.trend_signals, state.meta)   # noisy...
    assert np.all(np.abs(state.trend_signals - state.meta) < 1.0)  # ...but tethered


def test_same_seed_same_cycle():
    def run(seed):
        engine = seeded_engine(seed=seed)
        for day in range(CYCLE_TICKS):
            engine.step({0: day % N_ACTIONS}, {})
        s = engine.state
        return (round(s.followers, 6), len(s.history), s.flag_strikes,
                tuple(np.round(s.meta, 9)))
    assert run(7) == run(7)
    assert run(7) != run(8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_engine.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/engine.py
"""The ContentSim transition engine — the only module that advances the world."""
from typing import Any, Dict, List, Optional

import numpy as np
from rlgym.api import TransitionEngine

from domains.content.state import (
    AFFINITY_MAX, AFFINITY_MIN, AUTHORITY_BOOST, AUTHORITY_DECAY, AUTHORITY_GAIN,
    AUTHORITY_MAX, AUTHORITY_SKIP_DECAY, DRIFT_STD, F0, FATIGUE_DECAY,
    FOLLOW_RATE, JUMP_PROB, MAX_STRIKES, MIN_PRIOR_POSTS, N_FORMATS, N_TOPICS,
    OUTLIER_MULTIPLIER, REACH_EXPONENT, REPEAT_FLAG_STREAK, SIGNAL_NOISE_STD,
    STRIKE_PROB, V0, VIEW_NOISE_SIGMA, ContentState, PostRecord, decode_action,
)


class ContentEngine(TransitionEngine[int, ContentState, int]):
    """One `step` = one simulated day. Engine action is a plain int in [0, 37)."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self._state: ContentState = self.create_base_state()
        self._config: Dict[str, Any] = {}

    @property
    def agents(self) -> List[int]:
        return [0]

    @property
    def max_num_agents(self) -> int:
        return 1

    @property
    def state(self) -> ContentState:
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        self._config = value

    def create_base_state(self) -> ContentState:
        return ContentState()

    def set_state(self, desired_state: ContentState, shared_info: Dict[str, Any]) -> ContentState:
        self._state = desired_state
        return self._state

    def step(self, actions: Dict[int, int], shared_info: Dict[str, Any]) -> ContentState:
        state = self._state
        state.views_today = 0.0
        state.follows_today = 0.0
        state.outlier_today = False
        state.posted_today = False

        self._apply_action(int(actions[0]))
        self._roll_flag()
        self._drift()
        self._decay()
        self._refresh_signals()
        state.ticks += 1
        return state

    def close(self) -> None:
        pass

    # -- internals ------------------------------------------------------------

    def _apply_action(self, action: int) -> None:
        state = self._state
        decoded = decode_action(action)
        if decoded is None:
            state.last_action = None
            state.repeat_streak = 0
            return
        if decoded == state.last_action:
            state.repeat_streak += 1
        else:
            state.last_action = decoded
            state.repeat_streak = 1

        format_index, hook_index, topic_index = decoded
        quality = (state.meta[format_index]
                   * state.topic_heat[topic_index]
                   * state.hook_fit[format_index, hook_index]
                   * (1.0 + AUTHORITY_BOOST * state.authority[topic_index])
                   / (1.0 + state.fatigue[format_index]))
        reach = V0 * (state.followers / F0) ** REACH_EXPONENT
        noise = float(np.exp(self.rng.normal(0.0, VIEW_NOISE_SIGMA)))
        views = reach * quality * noise
        follows = FOLLOW_RATE * views * min(quality, 2.0)

        prior_views = [post.views for post in state.history]
        if (len(prior_views) >= MIN_PRIOR_POSTS
                and views >= OUTLIER_MULTIPLIER * float(np.median(prior_views))):
            state.outlier_today = True
            state.had_outlier = True

        state.history.append(PostRecord(
            format=format_index, hook=hook_index, topic=topic_index,
            day=state.ticks, views=views, follows=follows,
        ))
        state.followers += follows
        state.views_today = views
        state.follows_today = follows
        state.posted_today = True
        state.fatigue[format_index] += 1.0
        state.authority[topic_index] = min(
            AUTHORITY_MAX, state.authority[topic_index] + AUTHORITY_GAIN)

    def _roll_flag(self) -> None:
        state = self._state
        if state.restricted:
            return
        if (state.repeat_streak >= REPEAT_FLAG_STREAK
                and self.rng.random() < STRIKE_PROB):
            state.flag_strikes += 1
            if state.flag_strikes >= MAX_STRIKES:
                state.restricted = True

    def _drift(self) -> None:
        state = self._state
        state.meta = np.clip(
            state.meta + self.rng.normal(0.0, DRIFT_STD, N_FORMATS),
            AFFINITY_MIN, AFFINITY_MAX)
        state.topic_heat = np.clip(
            state.topic_heat + self.rng.normal(0.0, DRIFT_STD, N_TOPICS),
            AFFINITY_MIN, AFFINITY_MAX)
        if self.rng.random() < JUMP_PROB:
            jumped_format = int(self.rng.integers(N_FORMATS))
            state.meta[jumped_format] = float(
                self.rng.uniform(AFFINITY_MIN, AFFINITY_MAX))

    def _decay(self) -> None:
        state = self._state
        state.fatigue *= FATIGUE_DECAY
        decay = AUTHORITY_DECAY if state.posted_today else AUTHORITY_SKIP_DECAY
        state.authority *= decay

    def _refresh_signals(self) -> None:
        state = self._state
        state.trend_signals = np.clip(
            state.meta + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_FORMATS),
            0.0, AFFINITY_MAX)
        state.heat_signals = np.clip(
            state.topic_heat + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_TOPICS),
            0.0, AFFINITY_MAX)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_content_engine.py -v`
Expected: 12 passed. Full suite: 124 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content/engine.py tests/test_content_engine.py
git commit -m "feat: add ContentEngine with drifting meta, fatigue, and flag strikes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Mutator and done conditions

**Files:**
- Create: `domains/content/mutators.py`, `domains/content/dones.py`
- Test: `tests/test_content_mutators_dones.py`

**Interfaces:**
- Produces: `CycleStartMutator(rng=None)` (fresh 30-day cycle: log-uniform followers, random hidden tables, everything else zeroed); `RestrictionCondition` (termination), `CycleEndCondition(cycle_ticks=CYCLE_TICKS)` (truncation). Done conditions use the list-in/dict-out `is_done` signature.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_mutators_dones.py
import numpy as np
from domains.content.state import (
    ContentState, PostRecord, CYCLE_TICKS,
    INIT_FOLLOWERS_MIN, INIT_FOLLOWERS_MAX,
    AFFINITY_INIT_MIN, AFFINITY_INIT_MAX,
)
from domains.content.mutators import CycleStartMutator
from domains.content.dones import RestrictionCondition, CycleEndCondition


def test_cycle_start_randomizes_the_hidden_tables():
    state = ContentState()
    CycleStartMutator(rng=np.random.default_rng(0)).apply(state, {})
    assert INIT_FOLLOWERS_MIN <= state.followers <= INIT_FOLLOWERS_MAX
    for table in (state.meta, state.topic_heat, state.hook_fit.ravel()):
        assert np.all(table >= AFFINITY_INIT_MIN)
        assert np.all(table <= AFFINITY_INIT_MAX)
    # Signals are seeded from the fresh hidden state, not stale defaults.
    assert np.all(state.trend_signals >= 0.0)


def test_cycle_start_clears_a_dirty_state():
    state = ContentState()
    state.ticks, state.flag_strikes, state.restricted = 29, 3, True
    state.had_outlier, state.repeat_streak = True, 5
    state.last_action = (1, 1, 1)
    state.fatigue[:] = 3.0
    state.authority[:] = 1.0
    state.history = [PostRecord(format=0, hook=0, topic=0, day=1,
                                views=9.0, follows=1.0)]
    state.views_today, state.follows_today = 100.0, 5.0
    state.outlier_today, state.posted_today = True, True
    CycleStartMutator(rng=np.random.default_rng(1)).apply(state, {})
    assert state.ticks == 0 and state.flag_strikes == 0
    assert not state.restricted and not state.had_outlier
    assert state.history == [] and state.repeat_streak == 0
    assert state.last_action is None
    assert np.all(state.fatigue == 0.0) and np.all(state.authority == 0.0)
    assert state.views_today == 0.0 and state.follows_today == 0.0
    assert not state.outlier_today and not state.posted_today


def test_cycles_vary_between_resets():
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(10):
        state = ContentState()
        CycleStartMutator(rng=rng).apply(state, {})
        seen.add((round(state.followers, 3), tuple(np.round(state.meta, 6))))
    assert len(seen) > 1


def test_restriction_condition():
    condition = RestrictionCondition()
    clean, flagged = ContentState(), ContentState()
    flagged.restricted = True
    condition.reset([0], clean, {})
    assert condition.is_done([0], clean, {}) == {0: False}
    assert condition.is_done([0], flagged, {}) == {0: True}


def test_cycle_end_condition_fires_at_the_limit():
    condition = CycleEndCondition()
    condition.reset([0], ContentState(), {})
    early, done = ContentState(), ContentState()
    early.ticks, done.ticks = CYCLE_TICKS - 1, CYCLE_TICKS
    assert condition.is_done([0], early, {}) == {0: False}
    assert condition.is_done([0], done, {}) == {0: True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_mutators_dones.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/mutators.py
"""State mutators — the curriculum hook. A TrendShockMutator or
ColdStartMutator slots into a MutatorSequence here later."""
from typing import Any, Dict, Optional

import numpy as np
from rlgym.api import StateMutator

from domains.content.state import (
    AFFINITY_INIT_MAX, AFFINITY_INIT_MIN, INIT_FOLLOWERS_MAX, INIT_FOLLOWERS_MIN,
    N_FORMATS, N_HOOKS, N_TOPICS, SIGNAL_NOISE_STD, ContentState,
)


class CycleStartMutator(StateMutator[ContentState]):
    """Fresh 30-day cycle: new account size, new hidden tables, clean slate."""

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply(self, state: ContentState, shared_info: Dict[str, Any]) -> None:
        state.followers = float(np.exp(self.rng.uniform(
            np.log(INIT_FOLLOWERS_MIN), np.log(INIT_FOLLOWERS_MAX))))
        state.meta = self.rng.uniform(AFFINITY_INIT_MIN, AFFINITY_INIT_MAX, N_FORMATS)
        state.topic_heat = self.rng.uniform(AFFINITY_INIT_MIN, AFFINITY_INIT_MAX, N_TOPICS)
        state.hook_fit = self.rng.uniform(AFFINITY_INIT_MIN, AFFINITY_INIT_MAX,
                                          (N_FORMATS, N_HOOKS))
        # Day-0 signals: a noisy reading of the fresh hidden state, clipped like
        # the engine's, so the agent never starts blind.
        state.trend_signals = np.clip(
            state.meta + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_FORMATS),
            0.0, None)
        state.heat_signals = np.clip(
            state.topic_heat + self.rng.normal(0.0, SIGNAL_NOISE_STD, N_TOPICS),
            0.0, None)
        state.fatigue = np.zeros(N_FORMATS)
        state.authority = np.zeros(N_TOPICS)
        state.flag_strikes = 0
        state.restricted = False
        state.had_outlier = False
        state.history = []
        state.ticks = 0
        state.last_action = None
        state.repeat_streak = 0
        state.views_today = 0.0
        state.follows_today = 0.0
        state.outlier_today = False
        state.posted_today = False
```

```python
# domains/content/dones.py
"""RestrictionCondition TERMINATES (the account is dead; there is no tomorrow
worth valuing). CycleEndCondition TRUNCATES (the cycle ended, the account did
not — the critic must value the ongoing state: followers carry forward)."""
from typing import Any, Dict, List

from rlgym.api import DoneCondition

from domains.content.state import CYCLE_TICKS, ContentState


class RestrictionCondition(DoneCondition[int, ContentState]):
    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: ContentState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        dead = state.restricted
        return {agent: dead for agent in agents}


class CycleEndCondition(DoneCondition[int, ContentState]):
    def __init__(self, cycle_ticks: int = CYCLE_TICKS) -> None:
        self.cycle_ticks = cycle_ticks

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def is_done(self, agents: List[int], state: ContentState,
                shared_info: Dict[str, Any]) -> Dict[int, bool]:
        over = state.ticks >= self.cycle_ticks
        return {agent: over for agent in agents}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_content_mutators_dones.py -v`
Expected: 5 passed. Full suite: 129 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content/mutators.py domains/content/dones.py tests/test_content_mutators_dones.py
git commit -m "feat: add ContentSim cycle-start mutator and done conditions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Observation builder

**Files:**
- Create: `domains/content/obs.py`
- Test: `tests/test_content_obs.py`

**Interfaces:**
- Produces: `ContentObs(ObsBuilder[int, np.ndarray, ContentState, int])`, `OBS_SIZE = 49`, plus the layout constants baselines decode with: `FOLLOWERS_IDX = 0`, `DAY_COS_IDX = 1`, `DAY_SIN_IDX = 2`, `STRIKES_IDX = 3`, `TREND_SLICE = slice(4, 7)`, `HEAT_SLICE = slice(7, 10)`, `FATIGUE_SLICE = slice(10, 13)`, `AUTHORITY_SLICE = slice(13, 16)`, `HISTORY_START = 16`, `POST_DIMS = 11`.
  Layout: followers log-norm, day (cos, sin), strikes/3 = 4; trend signals (3) + heat signals (3), each `clip(x / AFFINITY_MAX, 0, 1)` = 6; fatigue `clip(x / MAX_FATIGUE_NORM, 0, 1)` (3) + authority (3) = 6; last K=3 posts most-recent-first × [format one-hot(3), hook one-hot(4), topic one-hot(3), `min(log1p(views)/log1p(VIEWS_REF), 1)`] = 33. Empty history slots all-zero. Hidden `meta`/`topic_heat`/`hook_fit` never read.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_obs.py
import numpy as np
from domains.content.state import (
    ContentState, PostRecord, AFFINITY_MAX, CYCLE_TICKS, HISTORY_K,
    MAX_STRIKES, VIEWS_REF,
)
from domains.content.obs import (
    AUTHORITY_SLICE, ContentObs, DAY_COS_IDX, DAY_SIN_IDX, FATIGUE_SLICE,
    FOLLOWERS_IDX, HEAT_SLICE, HISTORY_START, OBS_SIZE, POST_DIMS, STRIKES_IDX,
    TREND_SLICE,
)


def post(fmt=0, hook=0, topic=0, day=0, views=1000.0):
    return PostRecord(format=fmt, hook=hook, topic=topic, day=day,
                      views=views, follows=views * 0.01)


def test_obs_space_and_shape():
    builder = ContentObs()
    assert builder.get_obs_space(0) == OBS_SIZE == 49
    assert HISTORY_START + HISTORY_K * POST_DIMS == OBS_SIZE
    state = ContentState()
    builder.reset([0], state, {})
    obs = builder.build_obs([0], state, {})[0]
    assert obs.shape == (49,) and obs.dtype == np.float32


def test_empty_history_slots_are_zero():
    obs = ContentObs().build_obs([0], ContentState(), {})[0]
    assert np.all(obs[HISTORY_START:] == 0.0)


def test_slot_zero_is_the_most_recent_post():
    state = ContentState()
    state.history = [post(fmt=0, hook=0, topic=0, day=0),
                     post(fmt=2, hook=3, topic=1, day=1, views=VIEWS_REF)]
    obs = ContentObs().build_obs([0], state, {})[0]
    base = HISTORY_START
    assert obs[base + 2] == 1.0                     # format one-hot -> 2
    assert obs[base + 3 + 3] == 1.0                 # hook one-hot -> 3
    assert obs[base + 3 + 4 + 1] == 1.0             # topic one-hot -> 1
    assert np.isclose(obs[base + POST_DIMS - 1], 1.0)   # views at VIEWS_REF
    # Slot 1 is the older post (format 0).
    assert obs[base + POST_DIMS + 0] == 1.0


def test_signals_come_from_signals_not_hidden_state():
    state = ContentState()
    state.meta = np.array([2.0, 2.0, 2.0])
    state.trend_signals = np.array([0.5, 0.5, 0.5])
    obs = ContentObs().build_obs([0], state, {})[0]
    assert np.allclose(obs[TREND_SLICE], 0.5 / AFFINITY_MAX)


def test_strikes_and_authority_visible():
    state = ContentState()
    state.flag_strikes = 2
    state.authority = np.array([0.0, 1.0, 0.5])
    obs = ContentObs().build_obs([0], state, {})[0]
    assert np.isclose(obs[STRIKES_IDX], 2 / MAX_STRIKES)
    assert np.allclose(obs[AUTHORITY_SLICE], [0.0, 1.0, 0.5])


def test_day_encoding_is_continuous_at_the_cycle_boundary():
    builder = ContentObs()
    end_state, start_state = ContentState(), ContentState()
    end_state.ticks = CYCLE_TICKS - 1
    end = builder.build_obs([0], end_state, {})[0]
    start = builder.build_obs([0], start_state, {})[0]
    assert abs(end[DAY_COS_IDX] - start[DAY_COS_IDX]) < 0.25
    assert abs(end[DAY_SIN_IDX] - start[DAY_SIN_IDX]) < 0.25


def test_obs_stays_bounded_on_random_states():
    rng = np.random.default_rng(0)
    builder = ContentObs()
    for _ in range(200):
        state = ContentState()
        state.followers = float(rng.uniform(1.0, 1e7))
        state.ticks = int(rng.integers(0, CYCLE_TICKS))
        state.flag_strikes = int(rng.integers(0, MAX_STRIKES + 1))
        state.trend_signals = rng.uniform(0.0, AFFINITY_MAX, 3)
        state.heat_signals = rng.uniform(0.0, AFFINITY_MAX, 3)
        state.fatigue = rng.uniform(0.0, 10.0, 3)
        state.authority = rng.uniform(0.0, 1.0, 3)
        state.history = [post(fmt=int(rng.integers(3)), hook=int(rng.integers(4)),
                              topic=int(rng.integers(3)), day=d,
                              views=float(rng.uniform(0.0, 1e7)))
                         for d in range(int(rng.integers(0, 8)))]
        obs = builder.build_obs([0], state, {})[0]
        assert np.all(np.isfinite(obs))
        assert np.all(np.abs(obs) <= 1.0 + 1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_obs.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/obs.py
"""Observation builder: account status + noisy signals + recent-post history.

The hidden tables (meta, topic_heat, hook_fit) are NEVER read here — the agent
sees only trend_signals/heat_signals, the same way the real system only sees
its research corpus, not the algorithm's internals.
"""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ObsBuilder

from domains.content.state import (
    AFFINITY_MAX, CYCLE_TICKS, HISTORY_K, MAX_FATIGUE_NORM, MAX_FOLLOWERS_NORM,
    MAX_STRIKES, N_FORMATS, N_HOOKS, N_TOPICS, VIEWS_REF, ContentState,
)

FOLLOWERS_IDX = 0
DAY_COS_IDX, DAY_SIN_IDX = 1, 2
STRIKES_IDX = 3
TREND_SLICE = slice(4, 4 + N_FORMATS)
HEAT_SLICE = slice(4 + N_FORMATS, 4 + N_FORMATS + N_TOPICS)
FATIGUE_SLICE = slice(10, 10 + N_FORMATS)
AUTHORITY_SLICE = slice(13, 13 + N_TOPICS)
HISTORY_START = 16
POST_DIMS = N_FORMATS + N_HOOKS + N_TOPICS + 1   # one-hots + log-views norm
OBS_SIZE = HISTORY_START + HISTORY_K * POST_DIMS  # 49


class ContentObs(ObsBuilder[int, np.ndarray, ContentState, int]):
    """49 normalized dims. The day-of-cycle is encoded as (cos, sin): a raw tick
    count would present the cycle wrap as a cliff, the same discontinuity as a
    raw yaw angle in MiniRL or the hour-of-week in DispatchSim.
    """

    def get_obs_space(self, agent: int) -> int:
        return OBS_SIZE

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[int], state: ContentState,
                  shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[FOLLOWERS_IDX] = min(
            np.log1p(max(state.followers, 0.0)) / np.log1p(MAX_FOLLOWERS_NORM), 1.0)
        angle = 2.0 * np.pi * (state.ticks % CYCLE_TICKS) / CYCLE_TICKS
        obs[DAY_COS_IDX] = np.cos(angle)
        obs[DAY_SIN_IDX] = np.sin(angle)
        obs[STRIKES_IDX] = min(state.flag_strikes / MAX_STRIKES, 1.0)
        obs[TREND_SLICE] = np.clip(state.trend_signals / AFFINITY_MAX, 0.0, 1.0)
        obs[HEAT_SLICE] = np.clip(state.heat_signals / AFFINITY_MAX, 0.0, 1.0)
        obs[FATIGUE_SLICE] = np.clip(state.fatigue / MAX_FATIGUE_NORM, 0.0, 1.0)
        obs[AUTHORITY_SLICE] = np.clip(state.authority, 0.0, 1.0)

        offset = HISTORY_START
        for post in reversed(state.history[-HISTORY_K:]):   # most recent first
            obs[offset + post.format] = 1.0
            obs[offset + N_FORMATS + post.hook] = 1.0
            obs[offset + N_FORMATS + N_HOOKS + post.topic] = 1.0
            obs[offset + POST_DIMS - 1] = min(
                np.log1p(max(post.views, 0.0)) / np.log1p(VIEWS_REF), 1.0)
            offset += POST_DIMS
        return {agent: obs.copy() for agent in agents}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_content_obs.py -v`
Expected: 7 passed. Full suite: 136 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content/obs.py tests/test_content_obs.py
git commit -m "feat: add 49-dim content observation builder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Action parser

**Files:**
- Create: `domains/content/actions.py`
- Test: `tests/test_content_actions.py`

**Interfaces:**
- Produces: `ContentAction(ActionParser[int, int, int, ContentState, int])` re-exporting `N_ACTIONS = 37` and `SKIP_ACTION = 36` from the state module, plus `encode_action(format_index, hook_index, topic_index) -> int` (the inverse of `decode_action`, used by the scripted baselines). Pass-through parser: policy int → engine int, accepting numpy scalars.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_actions.py
import numpy as np
from domains.content.state import ContentState, decode_action
from domains.content.actions import (
    ContentAction, N_ACTIONS, SKIP_ACTION, encode_action,
)


def test_action_space_is_thirty_seven():
    assert N_ACTIONS == 37 and SKIP_ACTION == 36
    assert ContentAction().get_action_space(0) == 37


def test_encode_is_the_inverse_of_decode():
    for action in range(N_ACTIONS - 1):
        assert encode_action(*decode_action(action)) == action


def test_parse_is_identity_and_accepts_numpy():
    parser = ContentAction()
    parser.reset([0], ContentState(), {})
    assert parser.parse_actions({0: 7}, ContentState(), {}) == {0: 7}
    assert parser.parse_actions({0: np.int64(3)}, ContentState(), {}) == {0: 3}
    assert type(parser.parse_actions({0: np.int64(3)}, ContentState(), {})[0]) is int
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_actions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/actions.py
"""Action parser. Actions are already engine-shaped ints; this casts and
documents the space: a = format*(N_HOOKS*N_TOPICS) + hook*N_TOPICS + topic,
or SKIP_ACTION."""
from typing import Any, Dict, List

from rlgym.api import ActionParser

from domains.content.state import (
    N_ACTIONS, N_HOOKS, N_TOPICS, SKIP_ACTION, ContentState,
)

__all__ = ["ContentAction", "N_ACTIONS", "SKIP_ACTION", "encode_action"]


def encode_action(format_index: int, hook_index: int, topic_index: int) -> int:
    """(format, hook, topic) -> action int. Inverse of state.decode_action."""
    return format_index * (N_HOOKS * N_TOPICS) + hook_index * N_TOPICS + topic_index


class ContentAction(ActionParser[int, int, int, ContentState, int]):
    def get_action_space(self, agent: int) -> int:
        return N_ACTIONS

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[int, int], state: ContentState,
                      shared_info: Dict[str, Any]) -> Dict[int, int]:
        return {agent: int(action) for agent, action in actions.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_content_actions.py -v`
Expected: 3 passed. Full suite: 139 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content/actions.py tests/test_content_actions.py
git commit -m "feat: add content action parser over the 37-action space

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Reward functions

**Files:**
- Create: `domains/content/rewards.py`
- Test: `tests/test_content_rewards.py`

**Interfaces:**
- Produces: `ViewsReward`, `FollowGrowthReward`, `OutlierReward`, `RestrictionReward`, and a local `CombinedReward` (deliberate per-domain copy — domains are self-contained costumes, mirroring how `rlgym.rocket_league` and `domains/minirl` own their own). All use the 5-arg `get_rewards`.

**The ladder (wired in Task 8):** views 0.01 (dense), follow growth 0.1 (event), outlier 0.5 (event), restriction 10.0 (terminal). Components emit natural ranges; only `CombinedReward` applies weights.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_rewards.py
import numpy as np
import pytest
from domains.content.state import ContentState, VIEWS_REF, FOLLOWS_REF
from domains.content.rewards import (
    CombinedReward, FollowGrowthReward, OutlierReward, RestrictionReward,
    ViewsReward,
)

NOT_DONE = ({0: False}, {0: False})


def make_state(views=0.0, follows=0.0, outlier=False, restricted=False):
    state = ContentState()
    state.views_today = views
    state.follows_today = follows
    state.outlier_today = outlier
    state.restricted = restricted
    return state


def test_views_reward_is_log_normalized_and_zero_on_skip():
    reward = ViewsReward()
    reward.reset([0], make_state(), {})
    assert reward.get_rewards([0], make_state(views=0.0), *NOT_DONE, {})[0] == 0.0
    at_ref = reward.get_rewards([0], make_state(views=VIEWS_REF), *NOT_DONE, {})[0]
    assert at_ref == pytest.approx(1.0)
    tiny = reward.get_rewards([0], make_state(views=100.0), *NOT_DONE, {})[0]
    assert 0.0 < tiny < at_ref


def test_views_reward_saturates_on_viral_posts():
    reward = ViewsReward()
    reward.reset([0], make_state(), {})
    viral = reward.get_rewards([0], make_state(views=VIEWS_REF * 100), *NOT_DONE, {})[0]
    assert viral == pytest.approx(1.0)     # log-clip: variance must not swamp PPO


def test_follow_growth_reward_is_linear_in_follows():
    reward = FollowGrowthReward()
    reward.reset([0], make_state(), {})
    assert reward.get_rewards([0], make_state(follows=FOLLOWS_REF), *NOT_DONE, {})[0] \
        == pytest.approx(1.0)
    assert reward.get_rewards([0], make_state(follows=FOLLOWS_REF / 2), *NOT_DONE, {})[0] \
        == pytest.approx(0.5)


def test_outlier_reward_fires_on_the_day():
    reward = OutlierReward()
    reward.reset([0], make_state(), {})
    assert reward.get_rewards([0], make_state(), *NOT_DONE, {})[0] == 0.0
    assert reward.get_rewards([0], make_state(outlier=True), *NOT_DONE, {})[0] == 1.0


def test_restriction_reward_fires_negative():
    reward = RestrictionReward()
    reward.reset([0], make_state(), {})
    assert reward.get_rewards([0], make_state(), *NOT_DONE, {})[0] == 0.0
    assert reward.get_rewards([0], make_state(restricted=True),
                              {0: True}, {0: False}, {})[0] == -1.0


def test_combined_applies_weights_discriminably():
    # Components must evaluate to DIFFERENT values or swapped weights are
    # invisible (the exact test bug the MiniRL review caught).
    state = make_state(follows=FOLLOWS_REF / 2, restricted=True)  # follow=0.5, restriction=-1
    correct = CombinedReward((RestrictionReward(), 10.0), (FollowGrowthReward(), 0.1))
    correct.reset([0], make_state(), {})
    swapped = CombinedReward((RestrictionReward(), 0.1), (FollowGrowthReward(), 10.0))
    swapped.reset([0], make_state(), {})
    assert correct.get_rewards([0], state, {0: True}, {0: False}, {})[0] \
        == pytest.approx(-9.95)
    assert swapped.get_rewards([0], state, {0: True}, {0: False}, {})[0] \
        == pytest.approx(4.9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_rewards.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/rewards.py
"""Reward functions. Same ladder shape as MiniRL and the parked DispatchSim:
dense terms are training wheels and must never outweigh the terminal event.

Built-in hacking demo for Stage-3-style experiments: reward RAW views instead
of this ladder and watch the agent chase every trend signal into fatigue and
flag strikes — Goodhart with a ring light.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import RewardFunction

from domains.content.state import FOLLOWS_REF, VIEWS_REF, ContentState


class ViewsReward(RewardFunction[int, ContentState, float]):
    """Dense-ish: log(1+views_today)/log(1+VIEWS_REF), clipped at 1. Zero on
    skip days. The log + clip keeps the heavy-tailed noise from swamping PPO."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        value = min(np.log1p(max(state.views_today, 0.0)) / np.log1p(VIEWS_REF), 1.0)
        return {agent: float(value) for agent in agents}


class FollowGrowthReward(RewardFunction[int, ContentState, float]):
    """Event: follows gained today / FOLLOWS_REF. The compounding signal."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        return {agent: state.follows_today / FOLLOWS_REF for agent in agents}


class OutlierReward(RewardFunction[int, ContentState, float]):
    """Event: +1 the day a post clears 2x the trailing median — the same
    definition analyze/outliers.py uses in the real system."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        value = 1.0 if state.outlier_today else 0.0
        return {agent: value for agent in agents}


class RestrictionReward(RewardFunction[int, ContentState, float]):
    """Terminal: -1 when the account gets restricted. The episode ends with it."""

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: ContentState,
                    is_terminated: Dict[int, bool], is_truncated: Dict[int, bool],
                    shared_info: Dict[str, Any]) -> Dict[int, float]:
        value = -1.0 if state.restricted else 0.0
        return {agent: value for agent in agents}


class CombinedReward(RewardFunction[int, ContentState, float]):
    """Weighted sum. Deliberate per-domain copy: domains are self-contained."""

    def __init__(self, *weighted_rewards: Tuple[RewardFunction, float]) -> None:
        self.weighted_rewards = weighted_rewards

    def reset(self, agents: List[int], initial_state: ContentState,
              shared_info: Dict[str, Any]) -> None:
        for reward_fn, _ in self.weighted_rewards:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents: List[int], state: ContentState,
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

Run: `./.venv/bin/pytest tests/test_content_rewards.py -v`
Expected: 6 passed. Full suite: 145 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content/rewards.py tests/test_content_rewards.py
git commit -m "feat: add content reward ladder

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: build_env, episode events, and integration

**Files:**
- Create: `domains/content/env.py`
- Test: `tests/test_content_env.py`

**Interfaces:**
- Produces: `build_env(rng=None) -> RLGym`, `episode_events(state) -> {"outlier": bool, "restricted": bool}`, re-exported `OBS_SIZE`, `N_ACTIONS`, `EVENT_NAMES = ("outlier", "restricted")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_env.py
import numpy as np
from domains.content.env import (
    EVENT_NAMES, N_ACTIONS, OBS_SIZE, build_env, episode_events,
)
from domains.content.state import CYCLE_TICKS, SKIP_ACTION


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


def test_random_cycle_is_stable_and_finite():
    rng = np.random.default_rng(1)
    env = build_env(rng=rng)
    env.reset()
    steps = 0
    for _ in range(CYCLE_TICKS * 3):
        obs, rewards, terminated, truncated = env.step({0: int(rng.integers(N_ACTIONS))})
        assert np.all(np.isfinite(obs[0])) and np.isfinite(rewards[0])
        steps += 1
        if terminated[0] or truncated[0]:
            break
    assert steps <= CYCLE_TICKS


def test_spamming_one_post_ends_badly_or_times_out():
    env = build_env(rng=np.random.default_rng(2))
    env.reset()
    for _ in range(CYCLE_TICKS):
        obs, rewards, terminated, truncated = env.step({0: 0})
        if terminated[0]:
            assert env.state.restricted
            return
        if truncated[0]:
            return
    raise AssertionError("episode never ended")


def test_skipping_all_cycle_truncates_cleanly():
    env = build_env(rng=np.random.default_rng(3))
    env.reset()
    for _ in range(CYCLE_TICKS):
        obs, rewards, terminated, truncated = env.step({0: SKIP_ACTION})
        assert not terminated[0]                    # skipping can't get you banned
        if truncated[0]:
            assert env.state.ticks == CYCLE_TICKS
            return
    raise AssertionError("episode never truncated")


def test_episode_events_names_and_semantics():
    env = build_env(rng=np.random.default_rng(0))
    env.reset()
    events = episode_events(env.state)
    assert set(events) == set(EVENT_NAMES)
    assert events["outlier"] is False and events["restricted"] is False
    env.state.had_outlier = True
    assert episode_events(env.state)["outlier"] is True
    env.state.restricted = True
    assert episode_events(env.state)["restricted"] is True


def test_seeded_envs_reproduce():
    def run(seed):
        env = build_env(rng=np.random.default_rng(seed))
        env.reset()
        rewards = []
        for step in range(CYCLE_TICKS):
            _, r, term, trunc = env.step({0: step % N_ACTIONS})
            rewards.append(r[0])
            if term[0] or trunc[0]:
                break
        return rewards
    assert run(5) == run(5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_env.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/env.py
"""Assembles the six configuration objects into the ContentSim environment."""
from typing import Optional

import numpy as np
from rlgym.api import RLGym

from domains.content.actions import ContentAction, N_ACTIONS, SKIP_ACTION
from domains.content.dones import CycleEndCondition, RestrictionCondition
from domains.content.engine import ContentEngine
from domains.content.mutators import CycleStartMutator
from domains.content.obs import OBS_SIZE, ContentObs
from domains.content.rewards import (
    CombinedReward, FollowGrowthReward, OutlierReward, RestrictionReward,
    ViewsReward,
)
from domains.content.state import ContentState

__all__ = ["build_env", "episode_events", "OBS_SIZE", "N_ACTIONS", "EVENT_NAMES"]

EVENT_NAMES = ("outlier", "restricted")


def episode_events(state: ContentState) -> dict:
    """Episode-level events for core.rollout.VecEnv(event_fn=...)."""
    return {
        "outlier": state.had_outlier,
        "restricted": state.restricted,
    }


def build_env(rng: Optional[np.random.Generator] = None) -> RLGym:
    rng = rng if rng is not None else np.random.default_rng()
    return RLGym(
        state_mutator=CycleStartMutator(rng=rng),
        obs_builder=ContentObs(),
        action_parser=ContentAction(),
        reward_fn=CombinedReward(
            (ViewsReward(), 0.01),
            (FollowGrowthReward(), 0.1),
            (OutlierReward(), 0.5),
            (RestrictionReward(), 10.0),
        ),
        transition_engine=ContentEngine(rng=rng),
        termination_cond=RestrictionCondition(),
        truncation_cond=CycleEndCondition(),
    )
```

Note the single `rng` is shared by mutator and engine deliberately — one seed,
one cycle. Reproducibility comes from construction order, which is fixed.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_content_env.py -v`
Expected: 7 passed. Full suite: 152 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content/env.py tests/test_content_env.py
git commit -m "feat: assemble ContentSim environment with episode events

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Channel renderer

**Files:**
- Create: `domains/content/render.py`
- Test: `tests/test_content_render.py`

**Interfaces:**
- Produces: `draw_board(ax, state) -> None` — plugs into `core.dashboard.Dashboard(draw_fn=...)` unchanged. Daily post bars colored by format (log-view heights, outliers starred), strike pips, follower/day text, red wash on restriction.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_content_render.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_content_render.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# domains/content/render.py
"""Channel drawing. Knows about posts and strikes; nothing about training."""
import numpy as np
from matplotlib.patches import Rectangle

from domains.content.state import (
    CYCLE_TICKS, FORMAT_NAMES, MAX_STRIKES, VIEWS_REF, ContentState,
)

FORMAT_COLOURS = ("tab:blue", "tab:olive", "tab:purple")


def draw_board(ax, state: ContentState) -> None:
    """Draw one frame of the account's cycle onto `ax`. Clears the axes first."""
    ax.clear()
    ax.set_xlim(-0.5, CYCLE_TICKS - 0.5)
    ax.set_ylim(0.0, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])

    if state.restricted:
        ax.set_facecolor("#ffe5e5")
        ax.text(CYCLE_TICKS / 2, 1.25, "RESTRICTED",
                ha="center", color="tab:red", fontsize=12, fontweight="bold")
    else:
        ax.set_facecolor("white")

    median = float(np.median([p.views for p in state.history])) if state.history else 0.0
    for post in state.history:
        height = min(np.log1p(post.views) / np.log1p(VIEWS_REF), 1.0)
        ax.add_patch(Rectangle((post.day - 0.4, 0.0), 0.8, height,
                               facecolor=FORMAT_COLOURS[post.format],
                               edgecolor="0.3", linewidth=0.5))
        if median > 0.0 and post.views >= 2.0 * median:
            ax.plot([post.day], [height + 0.06], marker="*", markersize=10,
                    color="tab:orange")

    for strike in range(MAX_STRIKES):
        filled = strike < state.flag_strikes
        ax.plot([1 + strike], [1.2], marker="o", markersize=8,
                color=("tab:red" if filled else "0.85"))
    ax.text(CYCLE_TICKS - 1, 1.2,
            f"{state.followers:,.0f} followers | day {state.ticks}",
            ha="right", fontsize=9, va="center")
    for i, name in enumerate(FORMAT_NAMES):
        ax.text(0.5 + i * 9, 1.08, name, fontsize=7,
                color=FORMAT_COLOURS[i], va="center")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_content_render.py -v`
Expected: 3 passed. Full suite: 155 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add domains/content/render.py tests/test_content_render.py
git commit -m "feat: add content channel renderer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: train.py --domain flag and checkpointing

**Files:**
- Modify: `train.py`
- Test: `tests/test_train_smoke.py` (extend)

**Interfaces:**
- Produces: `run_training(..., domain: str = "minirl", save_path: Optional[str] = None)`; a `DOMAINS` registry mapping name → `dict(build_env, draw_fn, event_fn, event_names, obs_size, n_actions)`; CLI `--domain {minirl,content}` and `--save PATH`. On completion, saves `torch.save({"state_dict": ..., "obs_size": ..., "n_actions": ...}, save_path)` when `save_path` is set. `_record_episode` must use the domain's `build_env` (parameterize it — it currently hardcodes MiniRL's).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train_smoke.py`:

```python
def test_training_runs_on_the_content_domain(tmp_path):
    save = str(tmp_path / "content.pt")
    history = run_training(iterations=2, n_envs=2, horizon=64, render_every=0,
                           show_dashboard=False, seed=0, domain="content",
                           save_path=save)
    assert len(history) == 2
    for snapshot in history:
        assert "outlier_rate" in snapshot and "restricted_rate" in snapshot

    import torch
    payload = torch.load(save, weights_only=True)
    assert payload["obs_size"] == 49 and payload["n_actions"] == 37


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
from domains.content import env as content_env
from domains.content.render import draw_board

DOMAINS = {
    "minirl": dict(build_env=minirl_env.build_env, draw_fn=draw_arena,
                   event_fn=minirl_env.episode_events, event_names=("touch", "goal"),
                   obs_size=minirl_env.OBS_SIZE, n_actions=minirl_env.N_ACTIONS),
    "content": dict(build_env=content_env.build_env, draw_fn=draw_board,
                    event_fn=content_env.episode_events,
                    event_names=content_env.EVENT_NAMES,
                    obs_size=content_env.OBS_SIZE, n_actions=content_env.N_ACTIONS),
}
```

`run_training(iterations, n_envs, horizon, render_every, show_dashboard, seed,
domain="minirl", save_path=None)` looks everything up from `DOMAINS[domain]`,
builds `EpisodeTracker(spec["event_names"])`, passes `event_fn` into `VecEnv`,
sizes `ActorCritic(spec["obs_size"], spec["n_actions"])`, flattens with
`spec["obs_size"]`, and prints a generic line built from the snapshot.

**Regression-gate warning:** iterating `sorted(snapshot)` would print `goal`
BEFORE `touch`, changing the console format and breaking the byte-exact gate.
Keep the domain's event ordering by iterating `spec["event_names"]`
(insertion order), NOT `sorted(...)`:

```python
        rates = " | ".join(f"{name} {snapshot[f'{name}_rate']:5.2f}"
                           for name in spec["event_names"])
        print(f"iter {iteration:4d} | reward {snapshot['mean_episode_reward']:8.3f} "
              f"| {rates} | entropy {stats['entropy']:5.3f} "
              f"| episodes {snapshot['episodes']}")
```

With `event_names=("touch", "goal")` this reproduces the exact historical
format. `_record_episode(model, build_env_fn, max_steps=450)` takes the env
factory as a parameter. At the end:

```python
    if save_path is not None:
        torch.save({"state_dict": model.state_dict(),
                    "obs_size": spec["obs_size"],
                    "n_actions": spec["n_actions"]}, save_path)
```

CLI adds `--domain {minirl,content}` (default minirl) and `--save PATH`
(default None). The dashboard title becomes the domain name.

- [ ] **Step 4: Run tests and re-run the regression gate**

Run: `./.venv/bin/pytest tests/ -v`
Expected: **157 passed, 3 skipped**

Run: `./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0 | tail -1`
Expected — byte-for-byte the Task 1 Step 0 line:
`iter   12 | reward    0.172 | touch  0.13 | goal  0.00 | entropy 3.223 | episodes 100`

- [ ] **Step 5: Commit**

```bash
git add train.py tests/test_train_smoke.py
git commit -m "feat: multi-domain training with --domain flag and checkpointing

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: evaluate.py — the harness

**Files:**
- Create: `evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces:
  - `random_agent(obs, rng) -> int`; `make_consistent_agent(format_index=0, topic_index=0) -> agent_fn` (fixed format+topic, hook rotates daily — convergence without adaptation, and no identical-post streak); `make_trend_chaser_agent(rng) -> agent_fn` (argmax trend signal format, argmax heat signal topic, random hook — adaptation without convergence).
  - `evaluate(agent_fn, n_scenarios=200, seed=0) -> dict` with keys `n_scenarios`, `restriction_rate`, `outlier_rate` (share of scenarios with ≥1 outlier), `median_views` (median over every post across all scenarios), `mean_final_followers`, `mean_reward`, `worst` (list of `{"seed_index", "reward", "restricted"}` for the 5 worst scenarios). Scenario i uses `np.random.default_rng(np.random.SeedSequence(seed).spawn(n_scenarios)[i])` so every listed failure replays exactly.
  - `load_policy_agent(checkpoint_path) -> agent_fn` (argmax over `ActorCritic` logits).
  - CLI: `--agent {random,consistent,trend_chaser,policy} [--checkpoint PATH] [--scenarios N] [--seed N]`, prints a report table.
- `agent_fn` signature is `(obs: np.ndarray) -> int`; `random_agent` is wrapped at the call site to bind its rng. This is the interface an LLM-backed strategy agent drops into later.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
from evaluate import (
    evaluate, make_consistent_agent, make_trend_chaser_agent, random_agent,
)
from domains.content.actions import encode_action
from domains.content.obs import HEAT_SLICE, OBS_SIZE, TREND_SLICE
from domains.content.state import N_ACTIONS, SKIP_ACTION, decode_action


def test_evaluate_returns_the_report_shape():
    rng = np.random.default_rng(0)
    report = evaluate(lambda obs: int(rng.integers(N_ACTIONS)), n_scenarios=5, seed=0)
    assert report["n_scenarios"] == 5
    for key in ("restriction_rate", "outlier_rate", "median_views",
                "mean_final_followers", "mean_reward", "worst"):
        assert key in report
    assert 0.0 <= report["restriction_rate"] <= 1.0
    assert 0.0 <= report["outlier_rate"] <= 1.0
    assert len(report["worst"]) <= 5


def test_evaluate_is_reproducible():
    a = evaluate(lambda obs: SKIP_ACTION, n_scenarios=4, seed=3)
    b = evaluate(lambda obs: SKIP_ACTION, n_scenarios=4, seed=3)
    assert a == b


def test_consistent_agent_rotates_hooks_on_a_fixed_format_and_topic():
    agent = make_consistent_agent(format_index=1, topic_index=2)
    seen_hooks = set()
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for _ in range(8):
        decoded = decode_action(agent(obs))
        assert decoded is not None
        f, h, t = decoded
        assert f == 1 and t == 2
        seen_hooks.add(h)
    assert len(seen_hooks) == 4          # all hooks cycled -> no identical streak


def test_consistent_agent_never_gets_restricted():
    report = evaluate(make_consistent_agent(), n_scenarios=10, seed=0)
    assert report["restriction_rate"] == 0.0


def test_trend_chaser_follows_the_signals():
    rng = np.random.default_rng(0)
    agent = make_trend_chaser_agent(rng)
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[TREND_SLICE] = [0.1, 0.9, 0.2]
    obs[HEAT_SLICE] = [0.2, 0.1, 0.8]
    decoded = decode_action(agent(obs))
    assert decoded is not None
    f, h, t = decoded
    assert f == 1 and t == 2


def test_encode_used_by_baselines_matches_decode():
    assert decode_action(encode_action(2, 3, 1)) == (2, 3, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# evaluate.py
"""Run any (obs) -> action agent through seeded ContentSim scenarios.

This is the demo-asset half of the project: the same six objects that train a
policy also stress-test a finished strategy. Every scenario is seeded, so every
failure in the report replays exactly. An LLM-backed agent_fn that reads a
strategy.md is a drop-in — the interface already fits.
"""
import argparse
import itertools
from typing import Callable, Dict, List

import numpy as np
import torch

from core.ppo import ActorCritic
from domains.content.actions import encode_action
from domains.content.env import build_env
from domains.content.obs import HEAT_SLICE, TREND_SLICE
from domains.content.state import CYCLE_TICKS, N_ACTIONS, N_HOOKS

AgentFn = Callable[[np.ndarray], int]


def random_agent(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(N_ACTIONS))


def make_consistent_agent(format_index: int = 0, topic_index: int = 0) -> AgentFn:
    """Convergence without adaptation: one format, one topic, hooks rotating
    daily (identical posts would trip the flag mechanic — rotation is free)."""
    counter = itertools.count()

    def agent(obs: np.ndarray) -> int:
        hook_index = next(counter) % N_HOOKS
        return encode_action(format_index, hook_index, topic_index)
    return agent


def make_trend_chaser_agent(rng: np.random.Generator) -> AgentFn:
    """Adaptation without convergence: argmax the noisy signals every day."""

    def agent(obs: np.ndarray) -> int:
        format_index = int(np.argmax(obs[TREND_SLICE]))
        topic_index = int(np.argmax(obs[HEAT_SLICE]))
        hook_index = int(rng.integers(N_HOOKS))
        return encode_action(format_index, hook_index, topic_index)
    return agent


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
    all_views: List[float] = []
    for index in range(n_scenarios):
        env = build_env(rng=np.random.default_rng(seeds[index]))
        obs = env.reset()
        total_reward = 0.0
        for _ in range(CYCLE_TICKS):
            action = agent_fn(obs[0])
            obs, rewards, terminated, truncated = env.step({0: action})
            total_reward += rewards[0]
            if terminated[0] or truncated[0]:
                break
        state = env.state
        all_views.extend(post.views for post in state.history)
        results.append({
            "seed_index": index,
            "reward": round(total_reward, 4),
            "restricted": state.restricted,
            "had_outlier": state.had_outlier,
            "final_followers": state.followers,
        })
    report = {
        "n_scenarios": n_scenarios,
        "restriction_rate": sum(r["restricted"] for r in results) / n_scenarios,
        "outlier_rate": sum(r["had_outlier"] for r in results) / n_scenarios,
        "median_views": float(np.median(all_views)) if all_views else 0.0,
        "mean_final_followers": sum(r["final_followers"] for r in results) / n_scenarios,
        "mean_reward": round(sum(r["reward"] for r in results) / n_scenarios, 4),
        "worst": sorted(
            ({"seed_index": r["seed_index"], "reward": r["reward"],
              "restricted": r["restricted"]} for r in results),
            key=lambda r: r["reward"])[:5],
    }
    return report


def format_report(name: str, report: Dict) -> str:
    lines = [
        f"=== {name} over {report['n_scenarios']} scenarios ===",
        f"restriction rate     : {report['restriction_rate']:.2%}",
        f"outlier rate         : {report['outlier_rate']:.2%}",
        f"median views/post    : {report['median_views']:,.0f}",
        f"mean final followers : {report['mean_final_followers']:,.0f}",
        f"mean reward          : {report['mean_reward']:.3f}",
        "worst scenarios      : " + ", ".join(
            f"#{w['seed_index']}({w['reward']:.1f}{'!' if w['restricted'] else ''})"
            for w in report["worst"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an agent in ContentSim.")
    parser.add_argument("--agent",
                        choices=("random", "consistent", "trend_chaser", "policy"),
                        default="trend_chaser")
    parser.add_argument("--checkpoint", default="checkpoints/content.pt")
    parser.add_argument("--scenarios", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.agent == "random":
        rng = np.random.default_rng(args.seed)
        agent_fn: AgentFn = lambda obs: random_agent(obs, rng)
    elif args.agent == "consistent":
        agent_fn = make_consistent_agent()
    elif args.agent == "trend_chaser":
        agent_fn = make_trend_chaser_agent(np.random.default_rng(args.seed))
    else:
        agent_fn = load_policy_agent(args.checkpoint)

    print(format_report(args.agent, evaluate(agent_fn, args.scenarios, args.seed)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_evaluate.py -v`
Expected: 6 passed. Full suite: 163 passed, 3 skipped.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: add seeded evaluation harness with content baselines

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Train it, beat the baselines, hold the regression

**Files:**
- Create: `checkpoints/` (verify `.gitignore` covers it; add the entry if not)
- Test: none new; this task produces measured numbers.

**Interfaces:** consumes everything.

- [ ] **Step 1: Measure the baselines**

```bash
./.venv/bin/python evaluate.py --agent random --scenarios 200 --seed 0
./.venv/bin/python evaluate.py --agent consistent --scenarios 200 --seed 0
./.venv/bin/python evaluate.py --agent trend_chaser --scenarios 200 --seed 0
```

Record all numbers. Expected shape: `random` has a nonzero restriction rate
(occasional accidental streaks); `consistent` has zero restrictions and high
authority-driven followers; `trend_chaser` adapts but eats fatigue. If any
baseline is degenerate (e.g. restriction rate near 1.0, or all three
indistinguishable), STOP and re-examine the constants before training.

- [ ] **Step 2: Train the content policy**

```bash
./.venv/bin/python train.py --domain content --iterations 300 --render-every 0 --no-dashboard --seed 0 --save checkpoints/content.pt
```

Watch `restricted` fall toward zero and `reward` rise across iterations. Judge
over the full run, not the first 60 — the curve is non-monotonic early (MiniRL
lesson).

- [ ] **Step 3: Evaluate the trained policy on the SAME scenario seeds**

```bash
./.venv/bin/python evaluate.py --agent policy --checkpoint checkpoints/content.pt --scenarios 200 --seed 0
```

**Success criteria (from the spec):** policy beats `trend_chaser` on
`mean_final_followers` and beats `random` on every metric (for
`restriction_rate`, lower is better). If the policy loses: ONE tuning pass is
budgeted — adjust constants in `state.py` (`VIEW_NOISE_SIGMA`, `DRIFT_STD`,
`SIGNAL_NOISE_STD`) or reward weights in `env.py` only (no logic changes),
re-train, re-measure, and document what changed and why in the commit message.
If it still loses, STOP and report the numbers honestly rather than tuning in
circles.

- [ ] **Step 4: Hold the MiniRL regression one final time**

```bash
./.venv/bin/pytest tests/ -q
./.venv/bin/python train.py --iterations 12 --render-every 0 --no-dashboard --seed 0 | tail -1
```

Expected: 163 passed, 3 skipped; and the byte-exact iter-12 line from Task 1
Step 0.

- [ ] **Step 5: Commit the results**

```bash
git add -A
git commit -m "docs: record content training results and baseline comparison

<paste the four report tables into the commit body>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** core generalization + regression gate → Task 1 (adopted from
the shelved DispatchSim plan, with a Step 0 that re-captures the baseline line).
State/constants/action authority → Task 2. Engine (views model, drift, jumps,
flag strikes, outliers, fatigue/authority) → Task 3. Mutator + dones
(terminated vs truncated lesson) → Task 4. 49-dim obs with hidden-state
isolation → Task 5. 37-action space → Task 6. Reward ladder → Task 7 (with a
discriminating weights test — the MiniRL lesson pre-applied). `build_env` +
`event_fn` → Task 8. Renderer → Task 9. `--domain` + checkpoints → Task 10
(insertion-order rates preserve the byte-exact console format). Harness +
three baselines → Task 11. Training vs baselines + final regression → Task 12.
Curriculum mutators (`TrendShockMutator`, `ColdStartMutator`) are spec'd as
later hooks, deliberately not built — matching the spec's non-goals.

**Known deliberate choices:** `CombinedReward` duplicated per-domain (domains
are self-contained; mirrors `domains/minirl`'s own copy). One shared `rng`
between mutator and engine per env (one seed = one reproducible cycle). The
spec's obs table said "log-views z-score"; implemented as clipped normalized
log-views (`log1p(views)/log1p(VIEWS_REF)`) — simpler, bounded, same
information. `consistent` rotates hooks rather than repeating one action
verbatim because the flag mechanic punishes exact repetition by design; the
rotation IS the lesson. Engine RNG draw order is fixed (post noise → flag roll
→ drift → jump → signals) so seeded cycles reproduce.

**Type consistency:** `decode_action` defined once (Task 2), consumed by engine
(Task 3), inverse `encode_action` (Task 6) consumed by baselines (Task 11) and
tested against `decode_action` in both Task 6 and Task 11. `N_ACTIONS`/
`SKIP_ACTION` defined in state (Task 2), re-exported by actions (Task 6) and
env (Task 8), consumed by evaluate (Task 11). `EpisodeTracker(event_names)`
(Task 1) consumed by train.py (Task 10). Per-tick channels (`views_today`,
`follows_today`, `outlier_today`, `posted_today`) written by engine (Task 3),
read by rewards (Task 7). Obs layout constants (`TREND_SLICE`, `HEAT_SLICE`,
Task 5) consumed by trend_chaser (Task 11). Checkpoint payload keys (Task 10)
match `load_policy_agent` (Task 11).
