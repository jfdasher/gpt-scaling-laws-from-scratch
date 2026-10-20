import json
import math

import numpy as np
import pytest

from analysis import costs, isoflop, parametric, predict

TRUE = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)  # Hoffmann et al.; optimum lies inside our grid
BUDGETS = (3e15, 1e16, 3e16, 1e17)
RATIOS = (5.0, 10.0, 20.0, 40.0, 80.0)


def law(n, d):
    return TRUE["E"] + TRUE["A"] / n ** TRUE["alpha"] + TRUE["B"] / d ** TRUE["beta"]


def synthetic_results(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in BUDGETS:
        for r in RATIOS:
            n = math.sqrt(c / (6 * r))
            d = c / (6 * n)
            for s in (0, 1) if c == BUDGETS[0] else (0,):
                name = f"grid/{c:.0e}/r{r:g}" + ("/seed1" if s else "")
                out.append({"run_id": name, "name": name, "phase": "grid", "budget_flops": c, "n_params": n,
                            "tokens": d, "val_loss": law(n, d) + rng.normal(0, noise), "gpu_hours": 0.1,
                            "cost_usd": 0.4})
    return out


def test_isoflop_recovers_exponent():
    res = synthetic_results(noise=1e-4)
    optima = isoflop.isoflop_optima(isoflop.grid_points(res), sigma=1e-3, n_boot=200)
    assert all(o["interior"] for o in optima)
    exps = isoflop.scaling_exponents(optima)
    target = TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"])
    assert abs(exps["a"] - target) < 0.05
    assert math.isclose(exps["b"], 1 - exps["a"])
    assert exps["a_lo"] <= exps["a"] <= exps["a_hi"]


def test_seed_sigma():
    res = synthetic_results()
    res[1]["val_loss"] += 0.02  # seed1 of the first 3e15 point
    assert math.isclose(isoflop.seed_sigma(res), math.sqrt((0.02**2 / 5) / 2))


def test_parametric_recovers_law():
    res = synthetic_results(noise=1e-5)
    N, D, L = isoflop.grid_runs(res)
    fit = parametric.fit(N, D, L)
    assert abs(fit["alpha"] - TRUE["alpha"]) < 0.03
    assert abs(fit["beta"] - TRUE["beta"]) < 0.03
    assert abs(fit["E"] - TRUE["E"]) < 0.05


def test_prediction_and_compare(tmp_path):
    res = synthetic_results(noise=1e-4)
    rule = (math.log(3e-3) + 0.3 * math.log(2.5e6), -0.3)
    p = predict.make_prediction(res, rule, n_boot=50)
    assert {"isoflop", "parametric", "declared_config", "declared", "predicted_loss"} <= set(p)
    assert p["isoflop"]["n_opt_lo"] <= p["isoflop"]["n_opt"] <= p["isoflop"]["n_opt_hi"]
    out = tmp_path / "pred.json"
    predict.write_prediction(p, out)
    with pytest.raises(FileExistsError):
        predict.write_prediction(p, out)
    actual = {"val_loss": p["predicted_loss"]["parametric"]["value"] + 0.01, "name": "test/3e+17"}
    cmp = predict.compare(p, actual)
    assert math.isclose(cmp["parametric"]["error"], 0.01, abs_tol=1e-9)
    assert "inside_interval" in cmp["parametric"]


def test_costs():
    r = {"gpu_hours": 2.1, "cost_usd": 8.296, "phase": "grid", "name": "x", "run_id": "1"}
    assert costs.cost_line(r) == "this run: 2.10 GPU-hours, $8.30"
    failed = [{"phase": "dynamics", "name": "f", "run_id": "2", "forensic": {"gpu_seconds": 3600.0}}]
    totals = costs.phase_totals([r], failed, dollars_per_hour=4.0)
    assert totals["dynamics"]["gpu_hours"] == 1.0 and totals["dynamics"]["cost_usd"] == 4.0
    assert totals["grid"]["runs"] == 1
