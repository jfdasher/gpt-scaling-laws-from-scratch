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
    p = predict.make_prediction(res, rule, n_boot=50, manifest_path=tmp_path / "no_manifest.json")
    assert {"isoflop", "parametric", "declared_config", "declared", "predicted_loss"} <= set(p)
    assert p["isoflop"]["n_opt_lo"] <= p["isoflop"]["n_opt"] <= p["isoflop"]["n_opt_hi"]
    out = tmp_path / "pred.json"
    predict.write_prediction(p, out)
    with pytest.raises(FileExistsError):
        predict.write_prediction(p, out)
    actual = {"val_loss": p["predicted_loss"]["parametric"]["value"] + 0.01, "name": "test/3e+17",
              "run_id": predict.run_id(predict.TrainConfig(**p["declared_config"]))}
    cmp = predict.compare(p, actual)
    assert math.isclose(cmp["parametric"]["error"], 0.01, abs_tol=1e-9)
    assert "inside_param_interval" in cmp["parametric"]
    assert "inside_prediction_interval" in cmp["parametric"]
    assert math.isclose(cmp["parametric"]["error_over_sigma"], 0.01 / p["seed_sigma"], rel_tol=1e-9)
    assert p["predicted_loss"]["parametric"]["pred_lo"] <= p["predicted_loss"]["parametric"]["pred_hi"]


def test_costs():
    r = {"gpu_hours": 2.1, "cost_usd": 8.296, "phase": "grid", "name": "x", "run_id": "1"}
    assert costs.cost_line(r) == "this run: 2.10 GPU-hours, $8.30"
    failed = [{"phase": "dynamics", "name": "f", "run_id": "2", "forensic": {"gpu_seconds": 3600.0}}]
    totals = costs.phase_totals([r], failed, dollars_per_hour=4.0)
    assert totals["dynamics"]["gpu_hours"] == 1.0 and totals["dynamics"]["cost_usd"] == 4.0
    assert totals["grid"]["runs"] == 1


def test_gradient_matches_numeric():
    from scipy.optimize import check_grad

    N, D, L = isoflop.grid_runs(synthetic_results(noise=1e-3))
    args = (np.log(N), np.log(D), np.log(L), parametric.HUBER_DELTA)
    theta = np.array([6.0, 6.0, 0.5, 0.34, 0.28])
    err = check_grad(lambda t: parametric._objective(t, *args)[0], lambda t: parametric._objective(t, *args)[1], theta)
    assert err < 1e-3 * np.linalg.norm(parametric._objective(theta, *args)[1])


def test_optimal_allocation_matches_numeric():
    from scipy.optimize import minimize_scalar

    p = dict(TRUE)
    c = 3e15
    n, d = parametric.optimal_allocation(p, c)
    res = minimize_scalar(lambda x: law(math.exp(x), c / (6 * math.exp(x))), bounds=(math.log(1e5), math.log(1e9)),
                          method="bounded", options={"xatol": 1e-10})
    assert math.isclose(n, math.exp(res.x), rel_tol=0.01)
    assert math.isclose(n, 5.8e6, rel_tol=0.05)
    assert math.isclose(d, c / (6 * n))


def test_non_grid_phases_ignored():
    res = synthetic_results(noise=1e-4)
    base_pts, base_runs = isoflop.grid_points(res), isoflop.grid_runs(res)
    extra = [dict(res[0], phase="dynamics", val_loss=9.0, name="dyn/x"), dict(res[0], phase="lr", val_loss=7.0, name="lr/x")]
    assert isoflop.grid_points(res + extra) == base_pts
    for a, b in zip(isoflop.grid_runs(res + extra), base_runs):
        assert np.array_equal(a, b)


def test_run_ids_filter_excludes_other_runs():
    res = synthetic_results(noise=1e-4)
    keep = {r["run_id"] for r in res if r["budget_flops"] != 3e15}
    pts = isoflop.grid_points(res, run_ids=keep)
    assert 3e15 not in pts and set(pts) == set(isoflop.grid_points(res)) - {3e15}
    N, D, L = isoflop.grid_runs(res, run_ids=keep)
    assert len(N) < len(isoflop.grid_runs(res)[0])
    with pytest.raises(ValueError):
        isoflop.seed_sigma(res, run_ids=keep)  # seed pairs exist only at 3e15
    assert isoflop.seed_sigma(res, run_ids={r["run_id"] for r in res}) == isoflop.seed_sigma(res)


def test_grid_run_ids_versions():
    from analysis.common import grid_run_ids
    assert len(grid_run_ids("v2", "lr_rule.json")) == 25 and len(grid_run_ids("v1", "lr_rule.json")) == 25
    assert grid_run_ids("v1", "lr_rule.json") != grid_run_ids("v2", "lr_rule.json")


def _prediction(tmp_path, **kw):
    rule = (math.log(3e-3) + 0.3 * math.log(2.5e6), -0.3)
    return predict.make_prediction(synthetic_results(noise=1e-4), rule, n_boot=20, **kw)


def test_prediction_refuses_when_data_too_small(tmp_path):
    p = _prediction(tmp_path, manifest_path=tmp_path / "none.json")
    tokens = p["declared"]["tokens"]
    small = tmp_path / "val_manifest.json"
    small.write_text(json.dumps({"train_tokens": tokens}))  # needs tokens + 1
    with pytest.raises(ValueError, match="train tokens"):
        _prediction(tmp_path, manifest_path=small)
    small.write_text(json.dumps({"train_tokens": tokens + 1}))
    _prediction(tmp_path, manifest_path=small)


def test_compare_rejects_wrong_or_missing_test_run(tmp_path):
    p = _prediction(tmp_path, manifest_path=tmp_path / "none.json")
    with pytest.raises(ValueError, match="phase"):
        predict.select_test_result([{"phase": "grid", "run_id": "x"}])
    with pytest.raises(ValueError, match="deadbeef"):
        predict.compare(p, {"val_loss": 3.0, "run_id": "deadbeef"})


def test_compare_cli_refuses_overwrite(tmp_path):
    p = _prediction(tmp_path, manifest_path=tmp_path / "none.json")
    pred = tmp_path / "pred.json"
    predict.write_prediction(p, pred)
    results = tmp_path / "results"
    results.mkdir()
    rid = predict.run_id(predict.TrainConfig(**p["declared_config"]))
    (results / f"{rid}.json").write_text(json.dumps({"run_id": rid, "phase": "test", "name": "t", "val_loss": 3.5}))
    args = ["compare", "--prediction", str(pred), "--results", str(results)]
    predict.main(args)
    assert (tmp_path / "pred_outcome.json").exists()
    with pytest.raises(FileExistsError):
        predict.main(args)
    predict.main(args + ["--force"])
