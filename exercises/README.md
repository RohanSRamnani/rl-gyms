# Stage 3 exercises

The working implementations live in `domains/minirl/`. These stubs are the same
three files with the bodies removed. Reimplement them from understanding, then:

```bash
MINIRL_EXERCISES=1 ./.venv/bin/pytest tests/test_exercises.py -v
```

Peek at the reference whenever you're stuck — the point is to produce the code,
not to suffer.

## After the exercises: the breakage experiments

Predict the outcome first, then run it and see whether you were right.

1. In `domains/minirl/env.py`, raise `TouchReward`'s weight from `0.5` to `5.0`.
   Watch the agent farm touches instead of scoring while its reward curve climbs.
2. In `domains/minirl/env.py`, move `GoalCondition()` from `termination_cond` to
   the `truncation_cond` `AnyCondition`. Value learning should degrade.
3. In `domains/minirl/state.py`, change `ACTION_REPEAT` from `4` to `1` and then
   `16`. Observe the effect on how fast learning progresses.
4. In `domains/minirl/obs.py`, delete the two relative-position fields (and drop
   `OBS_SIZE` to 14). Learning should slow noticeably.

Use `git diff` to see what you changed and `git checkout --` to revert.
