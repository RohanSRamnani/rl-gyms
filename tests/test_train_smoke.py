import matplotlib
matplotlib.use("Agg")

from train import run_training


def test_training_runs_and_reports_metrics():
    history = run_training(iterations=3, n_envs=2, horizon=32,
                           render_every=0, show_dashboard=False, seed=0)
    assert len(history) == 3
    for snapshot in history:
        for key in ("mean_episode_reward", "touch_rate", "goal_rate", "episodes"):
            assert key in snapshot
