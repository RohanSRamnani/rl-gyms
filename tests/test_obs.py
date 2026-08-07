import numpy as np
from domains.minirl.state import Car, MiniRLState, SIDE_WALL_X, BACK_WALL_Y, CAR_MAX_SPEED
from domains.minirl.obs import DefaultObs


def make_state(car_pos=(0.0, -1000.0), ball_pos=(0.0, 0.0), is_orange=False,
               car_vel=(0.0, 0.0), ball_vel=(0.0, 0.0), yaw=0.0):
    car = Car(pos=np.array(car_pos, dtype=float), vel=np.array(car_vel, dtype=float),
              yaw=yaw, boost=100.0, is_orange=is_orange)
    return MiniRLState(ball_pos=np.array(ball_pos, dtype=float),
                       ball_vel=np.array(ball_vel, dtype=float), cars={0: car})


def test_obs_space_is_sixteen():
    assert DefaultObs().get_obs_space(0) == 16


def test_obs_has_expected_shape_and_dtype():
    builder = DefaultObs()
    state = make_state()
    builder.reset([0], state, {})
    obs = builder.build_obs([0], state, {})
    assert obs[0].shape == (16,)
    assert obs[0].dtype == np.float32


def test_obs_stays_in_bounds_across_random_states():
    rng = np.random.default_rng(0)
    builder = DefaultObs()
    for _ in range(500):
        state = make_state(
            car_pos=rng.uniform(-SIDE_WALL_X, SIDE_WALL_X, 2),
            ball_pos=rng.uniform(-SIDE_WALL_X, SIDE_WALL_X, 2),
            car_vel=rng.uniform(-CAR_MAX_SPEED, CAR_MAX_SPEED, 2),
            ball_vel=rng.uniform(-6000, 6000, 2),
            yaw=rng.uniform(-np.pi, np.pi),
        )
        obs = builder.build_obs([0], state, {})[0]
        assert np.all(np.abs(obs) <= 2.0), obs


def test_yaw_is_encoded_as_cos_sin_not_raw_angle():
    builder = DefaultObs()
    near_pi = builder.build_obs([0], make_state(yaw=np.pi - 1e-6), {})[0]
    near_neg_pi = builder.build_obs([0], make_state(yaw=-np.pi + 1e-6), {})[0]
    # A raw angle would jump by 2*pi here; cos/sin must be continuous.
    assert np.allclose(near_pi[4:6], near_neg_pi[4:6], atol=1e-4)


def test_orange_view_is_mirrored_so_both_teams_see_the_same_problem():
    builder = DefaultObs()
    blue = builder.build_obs([0], make_state(car_pos=(100.0, -1000.0), is_orange=False), {})[0]
    orange = builder.build_obs([0], make_state(car_pos=(-100.0, 1000.0), is_orange=True), {})[0]
    np.testing.assert_allclose(blue[0:2], orange[0:2], atol=1e-6)


def test_relative_ball_position_is_zero_when_on_top_of_the_ball():
    builder = DefaultObs()
    obs = builder.build_obs([0], make_state(car_pos=(500.0, 500.0), ball_pos=(500.0, 500.0)), {})[0]
    np.testing.assert_allclose(obs[11:13], [0.0, 0.0], atol=1e-9)
    assert np.isclose(obs[15], 0.0)


def test_reset_is_callable():
    builder = DefaultObs()
    builder.reset([0], make_state(), {})
