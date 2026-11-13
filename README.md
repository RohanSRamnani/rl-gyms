# RL Gyms

Hand-written PPO, and two reinforcement learning environments built on the
RLGym v2 API to train against it.

The point of the project is the boundary. `core/` is the learning machinery and
it never imports a domain. A domain is six composable objects (state, engine,
actions, observations, rewards, done conditions) plus a renderer. `train.py`
wires one into the other. Adding a new environment means writing those six
objects and nothing else, and a test fails the build if `core/` ever reaches
into `domains/`.

## What is in here

**`core/`** PPO from scratch: clipped surrogate objective, GAE, entropy bonus,
an actor-critic with sizes passed in as arguments so the file stays
domain-blind. Plus a vectorised rollout collector, an episode metrics tracker,
and a live matplotlib dashboard that draws the reward curve while it trains.

**`domains/minirl/`** A 2D top-down Rocket League. Real Rocket League units,
kickoff mutators, goal and touch rewards, no-touch timeouts. Class names match
their `rlgym.rocket_league` counterparts file for file, so the code reads
against the official examples.

**`domains/content/`** A content strategy simulator. One account, one tick per
day, thirty day episodes, and a hidden "meta" that drifts underneath the agent.
Rewards combine view counts, follower growth, outlier hits and a restriction
penalty, so the policy has to find what works now and keep finding it after it
stops working.

**`evaluate.py`** The part worth reading. It runs any `(obs) -> action`
function through seeded ContentSim scenarios, which means a trained policy and
a hand-written strategy go through the same gate. The scripted baselines are
built to isolate the two ways a content strategy fails:

- `consistent` converges but never adapts: one format, one topic, hooks rotating.
- `trend_chaser` adapts but never converges: argmax the noisy signals daily.

A policy that cannot beat both is not doing anything a rule could not. Every
scenario is seeded, so every failure in the report replays exactly.

**`exercises/`** The minirl observation, reward and done files with the bodies
removed, plus a set of predict-then-break experiments. Raise `TouchReward` from
0.5 to 5.0 and watch the agent farm touches while its reward curve climbs, which
is reward hacking you can see in about two minutes.

## Running it

```bash
python train.py --domain content --iterations 200
```

```bash
python evaluate.py
```

```bash
pytest
```

Python 3.13, `rlgym` 2.0.1, numpy, torch, matplotlib, pytest.

## Notes from building it

The RLGym v2 docs are wrong in two places that cost real time. `DoneCondition.is_done`
takes `(agents, state, shared_info)` and returns a dict keyed by agent, not what
the docs show. `RLGym.step()` returns a 4-tuple, `(obs, rewards, is_terminated,
is_truncated)`, with no `info`. `TransitionEngine` has no `reset`.

Refactors in here are gated on an exact training signature: twelve iterations at
a fixed seed must reproduce the same reward, touch, goal and entropy figures as
before the change. If the numbers move, the refactor changed behaviour, and that
is a bug rather than a style difference.

Design docs and implementation plans are in `docs/`.
