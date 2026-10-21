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
        plots.plot_frontier(optima, tmp_path / "frontier.png", predicted=(3e17, 3.7), actual=(3e17, 3.72),
                            parametric=(3e17, 3.75, 3.6, 3.9)),
        plots.plot_tokens_per_param(optima, tmp_path / "tpp.png"),
        plots.plot_lr_sweep([{"budget": 3e15, "ratio": 80.0, "n_params": 2.5e6, "lrs": [1e-3, 2e-3, 4e-3],
                              "losses": [4.1, 4.0, float("inf")], "best_lr": 2e-3}], tmp_path / "lr.png"),
        plots.plot_curves({"a": [{"step": s, "loss": 5 - s / 100, "grad_norm": 1.0} for s in range(100)],
                           "b": [{"step": s, "loss": 5 - s / 120, "grad_norm": 2.0} for s in range(100)]},
                          "test", tmp_path / "curves.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000


def test_plot_layer_diagnostics(tmp_path):
    rows = [{"step": s, **{f"grad_norm/layer_{i}": 1.0 + i * s for i in range(3)},
             **{f"attn_max_logit/layer_{i}": float("nan") if s == 40 else 5.0 * i for i in range(3)}}
            for s in range(0, 60, 20)]
    p = plots.plot_layer_diagnostics(rows, "diag", tmp_path / "layers.png")
    assert p.exists() and p.stat().st_size > 1000


def test_plot_dynamics_writes_files(tmp_path):
    import json

    res = tmp_path / "results"
    (res / "curves").mkdir(parents=True)
    names = ["grid/3e+16/r20", "dyn/lr3x", "dyn/lr3x-noclip", "dyn/lr10x", "dyn/lr10x-noclip", "dyn/lr10x-noclip+qknorm",
             "dyn/lr10x-noclip+zloss", "dyn/warmup0", "dyn/warmup1", "dyn/beta2-0.999", "dyn/batch32k", "dyn/batch128k",
             "dyn/batch256k", "dyn/batch512k"]
    for i, n in enumerate(names):
        rid = f"run{i:02d}"
        (res / f"{rid}.json").write_text(json.dumps({"run_id": rid, "name": n}))
        rows = [{"step": s, "loss": 6 - s / 50 + i * 0.01, "grad_norm": 1.0 + i} for s in range(100)]
        (res / "curves" / f"{rid}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        diag = [{"step": s, **{f"grad_norm/layer_{k}": 1.0 + k for k in range(2)},
                 **{f"attn_max_logit/layer_{k}": 3.0 * k for k in range(2)}} for s in range(0, 100, 20)]
        (res / "curves" / f"{rid}.diag.jsonl").write_text("\n".join(json.dumps(r) for r in diag))
    paths = plots.plot_dynamics(res, tmp_path / "figs")
    got = {p.name for p in paths}
    assert {"dyn_instability.png", "dyn_warmup.png", "dyn_beta2.png", "dyn_batch.png",
            "dyn_instability_layers_lr10x-noclip.png", "dyn_instability_layers_lr10x-noclip+qknorm.png",
            "dyn_instability_layers_lr10x-noclip+zloss.png", "dyn_instability_layers_lr3x.png"} == got
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
