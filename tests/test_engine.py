import numpy as np
import pytest
from domains.minirl.state import Car, MiniRLState, BACK_WALL_Y
from domains.minirl.engine import MiniRLEngine


def seeded_engine(car_pos=(0.0, -1000.0), ball_pos=(0.0, 0.0)):
    engine = MiniRLEngine()
    base = engine.create_base_state()
    base.cars = {0: Car(pos=np.array(car_pos, dtype=float), vel=np.zeros(2),
                        yaw=np.pi / 2, boost=100.0, is_orange=False)}
    base.ball_pos = np.array(ball_pos, dtype=float)
    engine.set_state(base, {})
    return engine


def test_agents_empty_before_any_state_is_set():
    engine = MiniRLEngine()
    assert engine.agents == []
    assert engine.max_num_agents == 2


def test_set_state_populates_agents():
    engine = seeded_engine()
    assert engine.agents == [0]
    assert engine.state.cars[0].boost == 100.0


def test_config_roundtrips():
    engine = MiniRLEngine()
    engine.config = {"foo": 1}
    assert engine.config == {"foo": 1}


def test_step_advances_ticks_by_number_of_rows():
    engine = seeded_engine()
    actions = {0: np.zeros((4, 3), dtype=np.float32)}
    engine.step(actions, {})
    assert engine.state.ticks == 4


def test_driving_forward_moves_the_car():
    engine = seeded_engine()
    actions = {0: np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (8, 1))}
    engine.step(actions, {})
    assert engine.state.cars[0].pos[1] > -1000.0


def test_step_stops_early_once_a_goal_is_scored():
    engine = seeded_engine(ball_pos=(0.0, BACK_WALL_Y - 50.0))
    engine.state.ball_vel = np.array([0.0, 5000.0])
    actions = {0: np.zeros((60, 3), dtype=np.float32)}
    engine.step(actions, {})
    assert engine.state.scoring_team == 0
    assert engine.state.ticks < 60


def test_create_base_state_is_fresh_each_call():
    engine = MiniRLEngine()
    a = engine.create_base_state()
    a.ticks = 99
    b = engine.create_base_state()
    assert b.ticks == 0


def test_close_is_safe():
    MiniRLEngine().close()
