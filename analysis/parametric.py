"""Parametric fit L(N, D) = E + A/N^alpha + B/D^beta (Hoffmann et al. approach 3), with run-level bootstrap.

    uv run python -m analysis.parametric results [--grid v1|v2]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from analysis import isoflop
from analysis.common import grid_run_ids, load_results

HUBER_DELTA = 1e-3
INIT_GRID = {"a": (0, 5, 10, 15, 20), "b": (0, 5, 10, 15, 20), "e": (-1, 0, 1),
             "alpha": (0.25, 0.5, 1.0, 1.5), "beta": (0.25, 0.5, 1.0, 1.5)}


def _objective(theta, ln_n, ln_d, ln_l, delta):
    """Huber loss on log residuals, scaled by 1/delta so its magnitude is O(n). Returns (value, gradient)."""
    a, b, e, alpha, beta = theta
    terms = np.stack([a - alpha * ln_n, b - beta * ln_d, np.full_like(ln_n, e)])
    pred = logsumexp(terms, axis=0)
    r = pred - ln_l
    small = np.abs(r) <= delta
    value = float(np.where(small, 0.5 * r**2 / delta, np.abs(r) - 0.5 * delta).sum())
    g = np.where(small, r / delta, np.sign(r))
    w = np.exp(terms - pred)  # softmax weights, shape (3, n)
    grad = np.array([(g * w[0]).sum(), (g * w[1]).sum(), (g * w[2]).sum(),
                     (-g * w[0] * ln_n).sum(), (-g * w[1] * ln_d).sum()])
    return value, grad


def _params(theta) -> dict:
    a, b, e, alpha, beta = (float(t) for t in theta)
    return {"E": math.exp(e), "A": math.exp(a), "B": math.exp(b), "alpha": alpha, "beta": beta,
            "a_implied": beta / (alpha + beta), "theta": [a, b, e, alpha, beta]}


def fit(N, D, L, inits=None, delta: float = HUBER_DELTA, keep: int = 20) -> dict:
    """Screen every start by objective value, then run L-BFGS-B from the best `keep` only."""
    ln_n, ln_d, ln_l = np.log(N), np.log(D), np.log(L)
    inits = inits if inits is not None else list(itertools.product(*INIT_GRID.values()))
    args = (ln_n, ln_d, ln_l, delta)
    starts = sorted(inits, key=lambda x0: _objective(np.array(x0, float), *args)[0])[:keep]
    best = None
    for x0 in starts:
        res = minimize(_objective, np.array(x0, float), args=args, jac=True, method="L-BFGS-B",
                       options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    out = _params(best.x)
    out["objective"] = float(best.fun)
    out["converged"] = bool(best.success)
    return out


def bootstrap(N, D, L, full: dict, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Resample runs with replacement; start each fit from the full fit plus a small grid."""
    rng = np.random.default_rng(seed)
    inits = [tuple(full["theta"])] + list(itertools.product((5, 10), (5, 10), (0,), (0.3, 0.6), (0.3, 0.6)))
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), len(N))
        out.append(fit(N[idx], D[idx], L[idx], inits=inits, keep=5))
    return out


def optimal_allocation(p: dict, budget: float) -> tuple[float, float]:
    alpha, beta, A, B = p["alpha"], p["beta"], p["A"], p["B"]
    g = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    n = g * (budget / 6) ** (beta / (alpha + beta))
    return n, budget / (6 * n)


def predicted_loss(p: dict, n: float, d: float) -> float:
    return p["E"] + p["A"] / n ** p["alpha"] + p["B"] / d ** p["beta"]


def interval(values) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values, float), [2.5, 97.5])
    return float(lo), float(hi)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--grid", choices=("v1", "v2"), default="v2")
    ap.add_argument("--lr-rule", default="lr_rule.json")
    args = ap.parse_args(argv)
    results_dir = args.results
    N, D, L = isoflop.grid_runs(load_results(results_dir), grid_run_ids(args.grid, args.lr_rule))
    full = fit(N, D, L)
    boots = bootstrap(N, D, L, full)
    summary = {"fit": full}
    for k in ("E", "A", "B", "alpha", "beta", "a_implied"):
        lo, hi = interval([b[k] for b in boots])
        summary[k] = {"value": full[k], "lo": lo, "hi": hi}
        print(f"{k:10} {full[k]:.4g} [{lo:.4g}, {hi:.4g}]")
    Path(results_dir, "fits").mkdir(parents=True, exist_ok=True)
    Path(results_dir, "fits", "parametric.json" if args.grid == "v2" else "parametric_v1.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
