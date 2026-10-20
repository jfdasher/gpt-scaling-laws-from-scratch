"""IsoFLOP method (Chinchilla approach 2): parabola in ln N per budget, then power laws in C.

    uv run python -m analysis.isoflop results
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.common import load_results


def grid_points(results) -> dict[float, list[tuple[float, float]]]:
    """budget -> sorted [(N, seed-averaged loss)]."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("phase") == "grid":
            groups[r["budget_flops"]][r["n_params"]].append(r["val_loss"])
    return {b: sorted((n, float(np.mean(ls))) for n, ls in by_n.items()) for b, by_n in groups.items()}


def grid_runs(results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, D, L) arrays, seed-averaged, for the parametric fit."""
    groups: dict = defaultdict(list)
    for r in results:
        if r.get("phase") == "grid":
            groups[(r["n_params"], r["tokens"])].append(r["val_loss"])
    keys = sorted(groups)
    return (np.array([k[0] for k in keys], float), np.array([k[1] for k in keys], float),
            np.array([np.mean(groups[k]) for k in keys]))


def seed_sigma(results) -> float:
    by_name = {r["name"]: r["val_loss"] for r in results if r.get("phase") == "grid"}
    diffs = [by_name[n] - by_name[n + "/seed1"] for n in by_name if n + "/seed1" in by_name]
    if not diffs:
        raise ValueError("no seed pairs found in grid results")
    return math.sqrt(float(np.mean(np.square(diffs))) / 2)


def _vertex(ln_n, loss):
    c2, c1, c0 = np.polyfit(ln_n, loss, 2)
    if c2 <= 0:
        return math.nan, math.nan, (c2, c1, c0)
    x = -c1 / (2 * c2)
    return x, c0 - c1**2 / (4 * c2), (c2, c1, c0)


def isoflop_optima(points_by_budget, sigma: float, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for budget in sorted(points_by_budget):
        pts = points_by_budget[budget]
        ln_n = np.log([p[0] for p in pts])
        loss = np.array([p[1] for p in pts])
        x, lmin, coeffs = _vertex(ln_n, loss)
        if math.isnan(x):
            raise ValueError(f"budget {budget:.0e}: loss vs ln N is not convex {coeffs}")
        boots = np.array([_vertex(ln_n, loss + rng.normal(0, sigma, loss.shape))[0] for _ in range(n_boot)])
        n_opt = math.exp(x)
        out.append({
            "budget": budget, "n_opt": n_opt,
            "n_opt_lo": math.exp(np.nanpercentile(boots, 2.5)), "n_opt_hi": math.exp(np.nanpercentile(boots, 97.5)),
            "d_opt": budget / (6 * n_opt), "tokens_per_param": budget / (6 * n_opt**2), "loss_min": lmin,
            "interior": bool(ln_n.min() < x < ln_n.max()), "frac_nonconvex_boot": float(np.isnan(boots).mean()),
            "points": pts, "parabola": [float(c) for c in coeffs], "_ln_n_boot": boots,
        })
    return out


def scaling_exponents(optima: list[dict]) -> dict:
    lnc = np.log([o["budget"] for o in optima])
    a, ln_k = np.polyfit(lnc, np.log([o["n_opt"] for o in optima]), 1)
    boots = np.stack([o["_ln_n_boot"] for o in optima])  # (budgets, n_boot)
    ok = np.all(np.isfinite(boots), axis=0)
    fits = np.array([np.polyfit(lnc, boots[:, j], 1) for j in np.flatnonzero(ok)])
    a_lo, a_hi = np.percentile(fits[:, 0], [2.5, 97.5])
    return {
        "a": float(a), "ln_k": float(ln_k), "a_lo": float(a_lo), "a_hi": float(a_hi),
        "b": float(1 - a), "b_lo": float(1 - a_hi), "b_hi": float(1 - a_lo),
        "note": "b = 1 - a exactly, because D_opt = C / (6 N_opt)",
        "_boot": fits,
    }


def predict_n_opt(exps: dict, budget: float) -> tuple[float, float, float]:
    lnc = math.log(budget)
    samples = np.exp(exps["_boot"][:, 1] + exps["_boot"][:, 0] * lnc)
    return math.exp(exps["ln_k"] + exps["a"] * lnc), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def frontier_predict(optima: list[dict], budget: float) -> float:
    """Straight line in (ln C, ln L_min) through the parabola minima; a point estimate with no floor term."""
    slope, icpt = np.polyfit(np.log([o["budget"] for o in optima]), np.log([o["loss_min"] for o in optima]), 1)
    return float(math.exp(icpt + slope * math.log(budget)))


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(results_dir="results") -> None:
    res = load_results(results_dir)
    sigma = seed_sigma(res)
    optima = isoflop_optima(grid_points(res), sigma)
    exps = scaling_exponents(optima)
    print(f"seed noise sigma = {sigma:.4f} nats")
    for o in optima:
        print(f"C={o['budget']:.0e} N_opt={o['n_opt']:,.0f} [{o['n_opt_lo']:,.0f}, {o['n_opt_hi']:,.0f}] "
              f"tok/param={o['tokens_per_param']:.1f} L_min={o['loss_min']:.4f} {'ok' if o['interior'] else 'EDGE'}")
    print(f"a = {exps['a']:.3f} [{exps['a_lo']:.3f}, {exps['a_hi']:.3f}]  b = {exps['b']:.3f}  ({exps['note']})")
    out = {"sigma": sigma, "optima": [public(o) for o in optima], "exponents": public(exps)}
    Path(results_dir, "fits").mkdir(exist_ok=True)
    Path(results_dir, "fits", "isoflop.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
