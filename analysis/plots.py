"""All figures. Every function takes plain data and an output path, and returns the path.

    uv run python -m analysis.plots results figures
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_isoflop(optima, path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for o in optima:
        n = np.array([p[0] for p in o["points"]])
        line = ax.plot(n, [p[1] for p in o["points"]], "o", label=f"C = {o['budget']:.0e}")[0]
        xs = np.linspace(np.log(n.min()) - 0.2, np.log(n.max()) + 0.2, 100)
        ax.plot(np.exp(xs), np.polyval(o["parabola"], xs), "-", color=line.get_color(), alpha=0.6)
        ax.plot(o["n_opt"], o["loss_min"], "*", color=line.get_color(), markersize=12)
    ax.set(xscale="log", xlabel="parameters N (total)", ylabel="final val loss (nats/token)", title="IsoFLOP profiles")
    ax.legend()
    return _save(fig, path)


def plot_optima(optima, exps, path, predicted=None, actual=None) -> Path:
    c = np.array([o["budget"] for o in optima])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cs = np.logspace(np.log10(c.min()) - 0.2, np.log10(max(c.max(), predicted[0] if predicted else 0)) + 0.2, 50)
    n_fit = np.exp(exps["ln_k"] + exps["a"] * np.log(cs))
    for ax, key, fitted, label, exp in (
        (axes[0], "n_opt", n_fit, "N_opt", exps["a"]),
        (axes[1], "d_opt", cs / (6 * n_fit), "D_opt", exps["b"]),
    ):
        y = np.array([o[key] for o in optima])
        if key == "n_opt":
            ax.errorbar(c, y, yerr=[y - [o["n_opt_lo"] for o in optima], [o["n_opt_hi"] for o in optima] - y], fmt="o")
        else:
            ax.plot(c, y, "o")
        ax.plot(cs, fitted, "-", label=f"fit: exponent {exp:.3f}")
        ref = fitted[0] * (cs / cs[0]) ** 0.5
        ax.plot(cs, ref, ":", color="gray", label="Chinchilla slope 0.5")
        if predicted and key == "n_opt":
            ax.plot(*predicted, "s", mfc="none", markersize=10, label="predicted 3e17")
        if actual and key == "n_opt":
            ax.plot(*actual, "s", markersize=7, label="trained 3e17")
        ax.set(xscale="log", yscale="log", xlabel="compute C (FLOPs)", ylabel=label)
        ax.legend(fontsize=8)
    return _save(fig, path)


def plot_frontier(optima, path, predicted=None, actual=None) -> Path:
    c = np.array([o["budget"] for o in optima])
    lmin = np.array([o["loss_min"] for o in optima])
    slope, icpt = np.polyfit(np.log(c), np.log(lmin), 1)
    cs = np.logspace(np.log10(c.min()) - 0.2, np.log10(max(c.max(), predicted[0] if predicted else 0)) + 0.2, 50)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c, lmin, "o", label="isoflop minima")
    ax.plot(cs, np.exp(icpt + slope * np.log(cs)), "-", label=f"L ∝ C^{slope:.3f}")
    if predicted:
        ax.plot(*predicted, "s", mfc="none", markersize=10, label="predicted 3e17")
    if actual:
        ax.plot(*actual, "s", markersize=7, label="trained 3e17")
    ax.set(xscale="log", yscale="log", xlabel="compute C (FLOPs)", ylabel="best val loss", title="Compute frontier")
    ax.legend()
    return _save(fig, path)


def plot_tokens_per_param(optima, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([o["budget"] for o in optima], [o["tokens_per_param"] for o in optima], "o-", label="measured optimum")
    ax.axhline(20, color="gray", ls=":", label="Chinchilla ≈ 20")
    ax.set(xscale="log", xlabel="compute C (FLOPs)", ylabel="D_opt / N_opt", title="Tokens per parameter at optimum")
    ax.legend()
    return _save(fig, path)


def plot_lr_sweep(groups, path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for g in groups:
        losses = [l if math.isfinite(l) else np.nan for l in g["losses"]]
        line = ax.plot(g["lrs"], losses, "o-", label=f"N={g['n_params']:,.0f} ({g['budget']:.0e})")[0]
        for lr, l in zip(g["lrs"], g["losses"]):
            if not math.isfinite(l):
                ax.plot(lr, np.nanmax(losses) if np.isfinite(np.nanmax(losses)) else 0, "x", color=line.get_color())
        ax.axvline(g["best_lr"], color=line.get_color(), ls=":", alpha=0.6)
    ax.set(xscale="log", xlabel="peak LR", ylabel="final val loss", title="LR sweep (x = diverged)")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_curves(curves: dict[str, list[dict]], title: str, path, metric: str = "loss", ylim=None) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, rows in curves.items():
        steps = [r["step"] for r in rows]
        axes[0].plot(steps, [r[metric] for r in rows], label=name, lw=1)
        axes[1].plot(steps, [r["grad_norm"] for r in rows], label=name, lw=1)
    axes[0].set(ylabel=metric, title=title)
    if ylim:
        axes[0].set_ylim(*ylim)
    axes[1].set(ylabel="grad norm (pre-clip)", xlabel="step", yscale="log")
    axes[0].legend(fontsize=8)
    return _save(fig, path)


def main(results_dir="results", out_dir="figures") -> None:
    fits = Path(results_dir, "fits", "isoflop.json")
    if fits.exists():
        iso = json.loads(fits.read_text())
        optima, exps = iso["optima"], iso["exponents"]
        pred_path, outcome_path = Path("predictions/3e17.json"), Path("predictions/3e17_outcome.json")
        pred = json.loads(pred_path.read_text()) if pred_path.exists() else None
        outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else None
        p_n = (pred["budget"], pred["isoflop"]["n_opt"]) if pred else None
        a_n = (pred["budget"], pred["declared"]["n_params"]) if pred and outcome else None
        p_l = (pred["budget"], pred["predicted_loss"]["isoflop_frontier"]["value"]) if pred else None
        a_l = (pred["budget"], outcome["actual_val_loss"]) if outcome else None
        plot_isoflop(optima, Path(out_dir, "isoflop.png"))
        plot_optima(optima, exps, Path(out_dir, "optima.png"), predicted=p_n, actual=a_n)
        plot_frontier(optima, Path(out_dir, "frontier.png"), predicted=p_l, actual=a_l)
        plot_tokens_per_param(optima, Path(out_dir, "tokens_per_param.png"))
    rule = Path("lr_rule.json")
    if rule.exists():
        plot_lr_sweep(json.loads(rule.read_text())["groups"], Path(out_dir, "lr_sweep.png"))
    print(f"figures written to {out_dir}/")


if __name__ == "__main__":
    main(*sys.argv[1:])
