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
