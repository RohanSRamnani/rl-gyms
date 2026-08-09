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


def test_training_runs_on_the_content_domain(tmp_path):
    save = str(tmp_path / "content.pt")
    history = run_training(iterations=2, n_envs=2, horizon=64, render_every=0,
                           show_dashboard=False, seed=0, domain="content",
                           save_path=save)
    assert len(history) == 2
    for snapshot in history:
        assert "outlier_rate" in snapshot and "restricted_rate" in snapshot

    import torch
    payload = torch.load(save, weights_only=True)
    assert payload["obs_size"] == 49 and payload["n_actions"] == 37


def test_minirl_stays_the_default():
    history = run_training(iterations=1, n_envs=2, horizon=32, render_every=0,
                           show_dashboard=False, seed=0)
    assert "touch_rate" in history[0]
