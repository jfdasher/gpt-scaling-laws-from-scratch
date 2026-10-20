"""Pre-registered prediction at 3e17, written and committed before the test run, then compared to it.

    uv run python -m analysis.predict make --results results --lr-rule lr_rule.json --out predictions/3e17.json
    uv run python -m analysis.predict compare --prediction predictions/3e17.json --results results
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analysis import isoflop, parametric
from analysis.common import load_results
from gptlab.train import TrainConfig, run_id
from gptlab.flops import param_count

TEST_BUDGET = 3e17
DATA_MANIFEST = "data/val_manifest.json"


def check_data_budget(tokens: int, manifest_path=DATA_MANIFEST) -> None:
    """The declared run needs tokens + 1 train tokens; refuse to pre-register a run the data cannot feed."""
    path = Path(manifest_path)
    if not path.exists():
        return
    have = json.loads(path.read_text()).get("train_tokens")
    if have is not None and tokens + 1 > have:
        raise ValueError(f"declared run needs {tokens + 1:,} train tokens but {path} has only {have:,}")


def make_prediction(results, rule, budget: float = TEST_BUDGET, n_boot: int = 1000, manifest_path=DATA_MANIFEST) -> dict:
    import sweep

    optima = isoflop.isoflop_optima(isoflop.grid_points(results), isoflop.seed_sigma(results), n_boot)
    exps = isoflop.scaling_exponents(optima)
    n_iso, n_lo, n_hi = isoflop.predict_n_opt(exps, budget)
    cfg = sweep.declared_test_config(n_iso, rule, budget)
    n_decl = param_count(cfg.n_layer, cfg.d_model)
    d_decl = cfg.total_steps * cfg.batch_tokens
    check_data_budget(d_decl, manifest_path)

    N, D, L = isoflop.grid_runs(results)
    full = parametric.fit(N, D, L)
    boots = parametric.bootstrap(N, D, L, full, n_boot=n_boot)
    n_par, d_par = parametric.optimal_allocation(full, budget)
    n_par_lo, n_par_hi = parametric.interval([parametric.optimal_allocation(b, budget)[0] for b in boots])
    sigma = isoflop.seed_sigma(results)
    mean_losses = [parametric.predicted_loss(b, n_decl, d_decl) for b in boots]
    loss_lo, loss_hi = parametric.interval(mean_losses)
    noise = np.random.default_rng(0).normal(0, sigma, len(mean_losses))
    pred_lo, pred_hi = parametric.interval(np.asarray(mean_losses) + noise)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "seed_sigma": sigma,
        "isoflop": {"n_opt": n_iso, "n_opt_lo": n_lo, "n_opt_hi": n_hi, "d_opt": budget / (6 * n_iso),
                    "a": exps["a"], "a_lo": exps["a_lo"], "a_hi": exps["a_hi"]},
        "parametric": {"n_opt": n_par, "n_opt_lo": n_par_lo, "n_opt_hi": n_par_hi, "d_opt": d_par,
                       "params": {k: full[k] for k in ("E", "A", "B", "alpha", "beta", "a_implied")}},
        "declared_config": asdict(cfg),
        "declared": {"n_params": n_decl, "tokens": d_decl, "rule": "isoflop N_opt snapped to a shape"},
        "predicted_loss": {
            "parametric": {"value": parametric.predicted_loss(full, n_decl, d_decl), "param_lo": loss_lo, "param_hi": loss_hi,
                           "pred_lo": pred_lo, "pred_hi": pred_hi},
            "isoflop_frontier": {"value": isoflop.frontier_predict(optima, budget), "note": "point estimate, no interval"},
        },
    }


def write_prediction(p: dict, path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; a pre-registered prediction is never overwritten silently")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, indent=2))


def select_test_result(results: list[dict]) -> dict:
    tests = [r for r in results if r.get("phase") == "test"]
    if not tests:
        raise ValueError("no result with phase == 'test' found; has the 3e17 run finished and been pulled?")
    return tests[0]


def compare(p: dict, actual: dict) -> dict:
    declared = run_id(TrainConfig(**p["declared_config"]))
    if actual.get("run_id") != declared:
        raise ValueError(f"test result run_id {actual.get('run_id')} != declared config run_id {declared}")
    out = {"actual_val_loss": actual["val_loss"], "run": actual.get("name")}
    for method, pred in p["predicted_loss"].items():
        entry = {"predicted": pred["value"], "error": actual["val_loss"] - pred["value"],
                 "rel_error": (actual["val_loss"] - pred["value"]) / pred["value"]}
        if "param_lo" in pred:
            entry["inside_param_interval"] = pred["param_lo"] <= actual["val_loss"] <= pred["param_hi"]
            entry["inside_prediction_interval"] = pred["pred_lo"] <= actual["val_loss"] <= pred["pred_hi"]
        if "seed_sigma" in p:
            entry["error_over_sigma"] = entry["error"] / p["seed_sigma"]
        out[method] = entry
    return out


def main(argv=None) -> None:
    import sweep

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--results", default="results")
    m.add_argument("--lr-rule", default="lr_rule.json")
    m.add_argument("--out", default="predictions/3e17.json")
    m.add_argument("--force", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--prediction", default="predictions/3e17.json")
    c.add_argument("--results", default="results")
    c.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        p = make_prediction(load_results(args.results), sweep.load_lr_rule(args.lr_rule))
        p["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        write_prediction(p, args.out, args.force)
        print(json.dumps({k: p[k] for k in ("isoflop", "declared", "predicted_loss")}, indent=2))
    else:
        p = json.loads(Path(args.prediction).read_text())
        actual = select_test_result(load_results(args.results))
        out_path = Path(args.prediction).with_name(Path(args.prediction).stem + "_outcome.json")
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} exists; use --force to overwrite")
        out = compare(p, actual)
        out_path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
