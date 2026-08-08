import numpy as np
from domains.minirl.state import MiniRLState, SIDE_WALL_X, BACK_WALL_Y
from domains.minirl.mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence


def blank():
    return MiniRLState(ball_pos=np.zeros(2), ball_vel=np.zeros(2), cars={})


def test_fixed_team_size_creates_requested_cars():
    state = blank()
    FixedTeamSizeMutator(blue_size=1, orange_size=0).apply(state, {})
    assert list(state.cars.keys()) == [0]
    assert state.cars[0].is_orange is False


def test_fixed_team_size_supports_one_v_one():
    state = blank()
    FixedTeamSizeMutator(blue_size=1, orange_size=1).apply(state, {})
    assert list(state.cars.keys()) == [0, 1]
    assert state.cars[0].is_orange is False
    assert state.cars[1].is_orange is True


def test_kickoff_centres_the_ball():
    state = blank()
    MutatorSequence(FixedTeamSizeMutator(), KickoffMutator()).apply(state, {})
    np.testing.assert_allclose(state.ball_pos, [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(state.ball_vel, [0.0, 0.0], atol=1e-9)


def test_kickoff_spawns_blue_car_in_its_own_half_with_full_boost():
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = blank()
        MutatorSequence(FixedTeamSizeMutator(), KickoffMutator(rng=rng)).apply(state, {})
        car = state.cars[0]
        assert car.pos[1] < 0
        assert abs(car.pos[0]) < SIDE_WALL_X
        assert abs(car.pos[1]) < BACK_WALL_Y
        assert car.boost == 100.0
        assert -np.pi <= car.yaw <= np.pi


def test_kickoff_spawns_orange_car_in_the_other_half():
    rng = np.random.default_rng(1)
    for _ in range(50):
        state = blank()
        MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1), KickoffMutator(rng=rng)
        ).apply(state, {})
        assert state.cars[1].pos[1] > 0


def test_kickoff_positions_vary_between_resets():
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(20):
        state = blank()
        MutatorSequence(FixedTeamSizeMutator(), KickoffMutator(rng=rng)).apply(state, {})
        seen.add(tuple(np.round(state.cars[0].pos, 3)))
    assert len(seen) > 1


def test_kickoff_clears_stale_episode_fields():
    state = blank()
    state.ticks = 500
    state.scoring_team = 1
    state.last_touch_tick = 42
    MutatorSequence(FixedTeamSizeMutator(), KickoffMutator()).apply(state, {})
    assert state.ticks == 0
    assert state.scoring_team is None
    assert state.last_touch_tick is None
