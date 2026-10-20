"""Parametric fit L(N, D) = E + A/N^alpha + B/D^beta (Hoffmann et al. approach 3), with run-level bootstrap.

    uv run python -m analysis.parametric results
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from analysis import isoflop
from analysis.common import load_results

HUBER_DELTA = 1e-3
INIT_GRID = {"a": (0, 5, 10, 15, 20), "b": (0, 5, 10, 15, 20), "e": (-1, 0, 1),
             "alpha": (0.25, 0.5, 1.0, 1.5), "beta": (0.25, 0.5, 1.0, 1.5)}


def _huber(r, delta):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


def _objective(theta, ln_n, ln_d, ln_l, delta):
    a, b, e, alpha, beta = theta
    pred = logsumexp(np.stack([a - alpha * ln_n, b - beta * ln_d, np.full_like(ln_n, e)]), axis=0)
    return float(_huber(pred - ln_l, delta).sum())


def _params(theta) -> dict:
    a, b, e, alpha, beta = (float(t) for t in theta)
    return {"E": math.exp(e), "A": math.exp(a), "B": math.exp(b), "alpha": alpha, "beta": beta,
            "a_implied": beta / (alpha + beta), "theta": [a, b, e, alpha, beta]}


def fit(N, D, L, inits=None, delta: float = HUBER_DELTA) -> dict:
    ln_n, ln_d, ln_l = np.log(N), np.log(D), np.log(L)
    inits = inits if inits is not None else list(itertools.product(*INIT_GRID.values()))
    best = None
    for x0 in inits:
        res = minimize(_objective, np.array(x0, float), args=(ln_n, ln_d, ln_l, delta), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    out = _params(best.x)
    out["objective"] = float(best.fun)
    return out


def bootstrap(N, D, L, full: dict, n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """Resample runs with replacement; start each fit from the full fit plus a small grid."""
    rng = np.random.default_rng(seed)
    inits = [tuple(full["theta"])] + list(itertools.product((5, 10), (5, 10), (0,), (0.3, 0.6), (0.3, 0.6)))
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(N), len(N))
        out.append(fit(N[idx], D[idx], L[idx], inits=inits))
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


def main(results_dir="results") -> None:
    N, D, L = isoflop.grid_runs(load_results(results_dir))
    full = fit(N, D, L)
    boots = bootstrap(N, D, L, full)
    summary = {"fit": full}
    for k in ("E", "A", "B", "alpha", "beta", "a_implied"):
        lo, hi = interval([b[k] for b in boots])
        summary[k] = {"value": full[k], "lo": lo, "hi": hi}
        print(f"{k:10} {full[k]:.4g} [{lo:.4g}, {hi:.4g}]")
    Path(results_dir, "fits").mkdir(exist_ok=True)
    Path(results_dir, "fits", "parametric.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
