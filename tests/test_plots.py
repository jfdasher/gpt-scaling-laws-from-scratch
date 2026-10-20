import math

from analysis import plots


def _optima():
    return [{"budget": c, "n_opt": 1e6 * (c / 3e15) ** 0.5, "n_opt_lo": 0.9e6 * (c / 3e15) ** 0.5,
             "n_opt_hi": 1.1e6 * (c / 3e15) ** 0.5, "d_opt": c / (6e6 * (c / 3e15) ** 0.5),
             "tokens_per_param": 20.0, "loss_min": 4 - 0.1 * math.log10(c / 3e15),
             "points": [(n, 4 + (math.log(n) - math.log(1e6)) ** 2 * 0.05) for n in (5e5, 7e5, 1e6, 1.4e6, 2e6)],
             "parabola": [0.05, 0.0, 4.0], "interior": True} for c in (3e15, 1e16, 3e16, 1e17)]


def test_plots_write_files(tmp_path):
    optima = _optima()
    exps = {"a": 0.5, "ln_k": math.log(1e6) - 0.5 * math.log(3e15), "b": 0.5}
    paths = [
        plots.plot_isoflop(optima, tmp_path / "isoflop.png"),
        plots.plot_optima(optima, exps, tmp_path / "optima.png", predicted=(3e17, 1e7), actual=(3e17, 1.1e7)),
        plots.plot_frontier(optima, tmp_path / "frontier.png", predicted=(3e17, 3.7), actual=(3e17, 3.72)),
        plots.plot_tokens_per_param(optima, tmp_path / "tpp.png"),
        plots.plot_lr_sweep([{"budget": 3e15, "ratio": 80.0, "n_params": 2.5e6, "lrs": [1e-3, 2e-3, 4e-3],
                              "losses": [4.1, 4.0, float("inf")], "best_lr": 2e-3}], tmp_path / "lr.png"),
        plots.plot_curves({"a": [{"step": s, "loss": 5 - s / 100, "grad_norm": 1.0} for s in range(100)],
                           "b": [{"step": s, "loss": 5 - s / 120, "grad_norm": 2.0} for s in range(100)]},
                          "test", tmp_path / "curves.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
