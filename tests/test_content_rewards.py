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
