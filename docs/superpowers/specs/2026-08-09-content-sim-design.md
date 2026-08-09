# ContentSim — Stage 4: a creator-economy domain on the same six objects

**Date:** 2026-08-09
**Status:** Approved design, pending implementation plan
**Replaces:** `2026-08-08-dispatch-sim-design.md` as Stage 4 (dispatch spec shelved, not retired)
**Companion:** `AgenticAgency/docs/superpowers/specs/2026-08-09-atmika-reinforcement-gym-design.md`
(the real-world system this domain simulates)

## Purpose

Stage 4 tests the project's central claim — that `core/` is domain-agnostic — by
standing up a second domain beside `domains/minirl/`: a simulated TikTok account
where posting decisions are actions and audience response is the environment.

It does triple duty:

1. **Learning:** the six objects wrap a creator-economy business process — the
   transferable skill this project exists to teach.
2. **Proof:** `core/` must run the new domain with only the event-extraction
   generalization described below — and MiniRL must keep training bit-identically
   afterwards.
3. **Demo/eval asset:** the simulator doubles as an *evaluation harness* — run any
   content strategy (scripted rules today, an LLM reading a `strategy.md` later)
   through seeded 30-day scenarios and produce a failure report. This is also the
   future pressure-test rig for the real Atmika strategy-rewrite loop: the hidden
   drifting "meta" below is exactly the non-stationarity that Atmika's 30-day
   policy rewrite exists to fight.

## The core/ refactor: domain-supplied event extraction

Adopted verbatim from the DispatchSim spec (its analysis stands — the leaks are in
`core/rollout.py:52-53`, `core/metrics.py`, `core/dashboard.py`, where MiniRL's
`scoring_team` / `last_touch_tick` attributes and `touch_rate` / `goal_rate`
vocabulary are hardcoded):

`VecEnv` gains an `event_fn` argument — a callable the DOMAIN supplies:

```python
def event_fn(state) -> Dict[str, bool]:
    """Named episode-level events, read from live state before any auto-reset."""
```

- MiniRL's: `{"touch": state.last_touch_tick is not None, "goal": state.scoring_team is not None}`
- ContentSim's: `{"outlier": state.had_outlier, "restricted": state.restricted}`

`EpisodeTracker` and `Dashboard` become name-agnostic: the tracker records
`Dict[str, bool]` events and reports `<name>_rate` for each; the dashboard plots
whatever rate keys the snapshot contains. `train.py` wires MiniRL's `event_fn` in,
so all existing metric names survive unchanged.

**Regression gate (non-negotiable):** after this refactor, MiniRL's seed-0
250-iteration run must reproduce today's verified numbers exactly
(reward 1.133, touch 0.79, goal 0.08 at iter 250). The refactor moves code, not
behaviour.

## The domain: ContentSim

You run a brand TikTok account. Once a day you choose what to post — format, hook
formula, topic family — or skip. A hidden, drifting algorithm "meta" decides what
the audience rewards. Followers compound. Converging on one topic builds niche
authority; hammering one format into a shifting meta decays. Spamming identical
posts risks an unoriginal-content restriction that ends the episode.

### Time

| Constant | Value |
|---|---|
| 1 tick | 1 simulated day |
| `ACTION_REPEAT` | 1 (one post decision per day) |
| Episode length | 30 ticks = one strategy cycle → **truncation** |
| Account restriction | → **termination** |

Same terminated/truncated lesson as MiniRL and Dispatch, in creator clothes: the
cycle ending is not the account ceasing to exist (followers carry into "next
month" conceptually — the critic must bootstrap), while a restriction genuinely
ends the episode with nothing to bootstrap.

### State (`domains/content/state.py`)

```python
F, H, T = 3, 4, 3   # formats, hook formulas, topic families
HISTORY_K = 3       # posts visible to the agent

@dataclass
class PostRecord:
    format: int          # 0 slideshow, 1 video, 2 talking_head
    hook: int            # 0 curiosity-list, 1 POV, 2 question, 3 transformation
    topic: int           # 0 home-styling, 1 slow-living, 2 product-story
    day: int
    views: float
    follows: float

@dataclass
class ContentState:
    followers: float
    meta: np.ndarray          # HIDDEN (F,): per-format algorithm affinity, drifts daily
    topic_heat: np.ndarray    # HIDDEN (T,): per-topic demand, drifts daily
    hook_fit: np.ndarray      # HIDDEN (F, H): fixed per-episode — learnable within episode
    trend_signals: np.ndarray # OBSERVED (F,): meta + noise, regenerated daily ("research corpus")
    heat_signals: np.ndarray  # OBSERVED (T,): topic_heat + noise
    fatigue: np.ndarray       # (F,): repetition penalty, decays daily
    authority: np.ndarray     # (T,): niche authority, builds with consistency
    flag_strikes: int         # 0..3
    restricted: bool          # True → termination
    had_outlier: bool         # any post this episode ≥ 2× trailing median (event_fn)
    history: List[PostRecord] # all posts this episode (obs reads last HISTORY_K)
    ticks: int
```

The `fatigue` / `authority` pair encodes the real Atmika tension: convergence
compounds (recognizability, bingeable page) but the meta drifts underneath you —
the agent must balance exploiting what works against tracking where the
algorithm is going.

### TransitionEngine (`engine.py`)

Per tick:

1. **Resolve today's post** (if not skip):
   `views = reach(followers) × meta[f] × topic_heat[t] × hook_fit[f, h]
   × (1 + authority[t]) / (1 + fatigue[f]) × LogNormal(0, σ)` —
   `reach(followers) = V0 × (followers / F0)^0.6`. The heavy-tailed noise is
   load-bearing: it produces genuine outliers, which is what the reward ladder
   and the analyzer analog key on. `follows ∝ views × quality` (quality = the
   deterministic multipliers, a retention proxy); `followers += follows`;
   `fatigue[f] += 1`, `authority[t] += δ`. Skip → zero views, authority decays
   extra.
2. **Drift:** `meta` and `topic_heat` take a small clipped random-walk step; with
   small per-day probability a trend *jump* reshuffles one format's affinity.
3. **Flag mechanic:** posting the identical (format, hook, topic) 3+ consecutive
   days rolls a per-day strike probability; `flag_strikes == 3` →
   `restricted = True`. (The unoriginal-content classifier analog — the same
   force that makes the real system rewrite all words, §8.2 of the Atmika spec.)
4. **Fresh signals:** `trend_signals = meta + N(0, σ_obs)`, likewise
   `heat_signals`. All fatigue decays.

All randomness comes from the engine-owned seeded RNG — same pattern as
`KickoffMutator`. Deterministic given the seed.

### StateMutator (`mutators.py`)

`CycleStartMutator(rng)` — fresh cycle: followers log-uniform in [1k, 20k],
random hidden `meta` / `topic_heat` / `hook_fit`, zero fatigue/authority/strikes,
empty history. Curriculum hooks slot into a `MutatorSequence` later without
touching anything else: `TrendShockMutator` (guaranteed mid-episode jump),
`ColdStartMutator` (100 followers) — same trick as `KickoffMutator`.

### ObsBuilder (`obs.py`)

Fixed-size vector. Hidden state never leaks — the agent sees only the noisy
signals:

| Fields | Dims |
|---|---|
| followers (log-normalized), day-of-cycle (cos, sin), strikes / 3 | 4 |
| trend signal per format (F) + heat signal per topic (T) | 6 |
| fatigue per format (F) + authority per topic (T) | 6 |
| last K = 3 posts: format one-hot(3) + hook one-hot(4) + topic one-hot(3) + log-views z-score | 33 |
| **total** | **49** |

Empty history slots zero-padded. The (cos, sin) day encoding is the same
angle-discontinuity lesson as MiniRL's yaw and Dispatch's hour-of-week.

### ActionParser (`actions.py`)

Discrete lookup table: (format, hook, topic) = F×H×T = 36 post actions, plus
`skip` = **37 actions**. Every action is legal; a mismatched post into a cold
meta is its own punishment — the agent must *learn* the hidden tables from
observed views, which is the point.

### RewardFunction (`rewards.py`) — the ladder again

| Component | Weight | Type |
|---|---|---|
| `ViewsReward` (log(1 + views) / log(1 + V_ref) for today's post) | 0.01 | dense |
| `FollowGrowthReward` (follows gained / F_ref) | 0.1 | event |
| `OutlierReward` (+1 when views ≥ 2× trailing median of own posts — mirrors `analyze/outliers.py`) | 0.5 | event |
| `RestrictionReward` (−1 on restriction) | 10.0 | terminal |

Same three-orders-of-magnitude shape as MiniRL and Dispatch. The built-in
reward-hacking experiment: reward *raw views* instead of the ladder and watch the
agent chase every trend signal into fatigue and flag-strikes — Goodhart with a
ring light.

### DoneConditions (`dones.py`)

`RestrictionCondition` → termination. `CycleEndCondition(30)` → truncation.

### Renderer (`render.py`)

`draw_board(ax, state)`: follower-growth line, daily post bars colored by format
(outliers starred), trend-signal ribbons behind the bars, strike pips, red wash
on restriction. Plugs into `core/dashboard.py` unchanged via `draw_fn`.

## Evaluation-harness mode (the demo asset)

`evaluate.py` — runs **any policy** through M seeded 30-day scenarios, no
learning:

```python
def evaluate(agent_fn, n_scenarios=200, seed=0) -> report
# agent_fn: (obs: np.ndarray) -> int    — trained net, scripted rules, anything
```

Ships with three scripted baselines:

- `random` — uniform over 37 actions,
- `consistent` — the same sensible post every day (convergence without adaptation),
- `trend_chaser` — argmax trend signal daily, random hook/topic (adaptation
  without convergence),

and produces a table: median views/post, outlier rate, day-30 followers,
restriction rate, plus per-scenario worst cases with seeds so any failure replays
exactly.

This file is the pitch artifact. An LLM-backed `agent_fn` that reads a
`strategy.md` — pressure-testing Atmika's strategy-rewrite loop in simulation —
is a later drop-in; the interface already fits, but it is explicitly out of scope
now.

## Non-goals

- Real TikTok data, APIs, or scraping — the audience model is plausible, not
  calibrated.
- Language — posts are structs (format, hook, topic), not captions or scripts.
- Modeling the research engine as actions — trend signals arrive in the obs for
  free; a "research vs post" action trade-off is a possible later mutator, not
  built now.
- LLM integration (interface reserved, not built).
- Multi-account, cross-platform, monetization/conversion modeling.
- Self-play / multi-agent (the parked MiniRL findings stay parked).

## Success criteria

1. **Reuse proof:** `domains/content/` + `evaluate.py` + a thin `train.py` wiring
   change; `core/` diff is the event-extraction refactor only.
2. **Regression:** MiniRL seed-0 numbers reproduce exactly post-refactor.
3. **Learning:** trained policy beats `trend_chaser` on day-30 followers and
   beats `random` on everything, within ~10 min CPU.
4. **Harness:** `evaluate.py` report reproduces any listed failure from its seed.
5. All existing tests keep passing; new domain gets the same test discipline
   (physics-free, so: engine transitions, drift and jump behaviour, flag logic,
   follower compounding, obs bounds and padding, action decoding, done wiring —
   plus falsification-style tests for the reward ladder).

## Risks

- **The refactor regresses MiniRL silently.** Mitigated by the bit-identical
  seed-0 gate before any content code lands.
- **37 actions over hidden drifting state may be too hard for fast PPO
  convergence.** σ, drift rate, and jump probability are constants; tuning them
  with the dashboard until `trend_chaser` is beatable-but-not-free is an explicit
  budgeted pass — a teaching moment, same as Dispatch's planned tuning pass.
- **Heavy-tail noise swamps the learning signal.** Mitigated by log-views in the
  dense reward and a median (not mean) outlier baseline; if variance still
  dominates, lower σ before touching the ladder.
