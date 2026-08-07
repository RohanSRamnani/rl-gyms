import numpy as np
from domains.minirl.state import MiniRLState, ACTION_REPEAT
from domains.minirl.actions import LookupTableAction, RepeatAction, N_ACTIONS


def blank():
    return MiniRLState(ball_pos=np.zeros(2), ball_vel=np.zeros(2), cars={})


def test_action_space_is_thirty():
    assert N_ACTIONS == 30
    assert LookupTableAction().get_action_space(0) == 30


def test_every_action_maps_to_a_valid_control_triple():
    parser = LookupTableAction()
    for index in range(N_ACTIONS):
        control = parser.parse_actions({0: index}, blank(), {})[0]
        assert control.shape == (3,)
        assert control[0] in (-1.0, 0.0, 1.0)
        assert control[1] in (-1.0, -0.5, 0.0, 0.5, 1.0)
        assert control[2] in (0.0, 1.0)


def test_table_entries_are_unique():
    parser = LookupTableAction()
    seen = {tuple(parser.parse_actions({0: i}, blank(), {})[0]) for i in range(N_ACTIONS)}
    assert len(seen) == N_ACTIONS


def test_numpy_scalar_actions_are_accepted():
    parser = LookupTableAction()
    control = parser.parse_actions({0: np.int64(3)}, blank(), {})[0]
    assert control.shape == (3,)


def test_repeat_action_tiles_into_ticks():
    parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)
    engine_action = parser.parse_actions({0: 5}, blank(), {})[0]
    assert engine_action.shape == (ACTION_REPEAT, 3)
    for row in engine_action:
        np.testing.assert_allclose(row, engine_action[0])


def test_repeat_action_preserves_the_wrapped_action_space():
    parser = RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT)
    assert parser.get_action_space(0) == 30


def test_reset_is_callable():
    parser = RepeatAction(LookupTableAction())
    parser.reset([0], blank(), {})
