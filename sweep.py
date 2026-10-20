"""Run configs for every phase, the LR rule, and a dry-run cost planner. Never launches anything.

    uv run python sweep.py plan <phase> [--lr-rule lr_rule.json] [--from-run NAME]
    uv run python sweep.py fit-lr-rule [--results results]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from gptlab.flops import BATCH_TOKENS, H100_DOLLARS_PER_HOUR, choose_shape, param_count, size_for_budget
from gptlab.train import TrainConfig, run_id

PHASES = ("smoke", "lr", "grid", "dynamics", "dynamics-fix", "test")
GRID_BUDGETS = (3e15, 1e16, 3e16, 1e17)
GRID_RATIOS = (5.0, 10.0, 20.0, 40.0, 80.0)
GRID_EXTRA_RATIOS: dict[float, tuple[float, ...]] = {}  # phase-3 gate, e.g. {1e17: (160.0,)}
SEED1_OFFSET = 1_000_000_000
TEST_BUDGET = 3e17
SMOKE_PREEMPT_STEP = 100
# (budget, tokens/param, centre LR, number of LRs); the first two are the rule's ends, the third is the check
LR_SWEEP = ((3e15, 80.0, 4e-3, 5), (3e16, 5.0, 2e-3, 5), (1e16, 20.0, 3e-3, 3))
DYNAMICS_BUDGET, DYNAMICS_RATIO = 3e16, 20.0
DEFAULT_FLOPS_PER_S = 7e13  # until bench.json exists
RUN_OVERHEAD_S = 90.0  # container start + compile + final eval


def make_config(phase: str, name: str, budget: float, ratio: float, lr: float, **overrides) -> TrainConfig:
    s = size_for_budget(budget, ratio, overrides.get("batch_tokens", BATCH_TOKENS))
    return TrainConfig(
        name=name, phase=phase, n_layer=s.n_layer, d_model=s.d_model, total_steps=s.steps, peak_lr=lr,
        budget_flops=budget, tokens_per_param=ratio, **overrides,
    )


def lr_values(center: float, n: int) -> tuple[float, ...]:
    return tuple(float(f"{center * 2.0**k:.4g}") for k in range(-(n // 2), n // 2 + 1))


def load_lr_rule(path="lr_rule.json") -> tuple[float, float]:
    d = json.loads(Path(path).read_text())
    return d["c0"], d["c1"]


def lr_for(n_params: int, rule: tuple[float, float]) -> float:
    c0, c1 = rule
    return float(f"{math.exp(c0 + c1 * math.log(n_params)):.4g}")


def smoke_configs() -> list[TrainConfig]:
    base = TrainConfig(name="smoke/a", phase="smoke", n_layer=4, d_model=256, total_steps=200, peak_lr=2e-3, eval_every=50)
    return [base, replace(base, name="smoke/preempt", tag="preempt")]


def lr_configs() -> list[TrainConfig]:
    return [
        make_config("lr", f"lr/{budget:.0e}/r{ratio:g}/lr{lr:.3g}", budget, ratio, lr)
        for budget, ratio, center, n in LR_SWEEP
        for lr in lr_values(center, n)
    ]


def grid_configs(rule) -> list[TrainConfig]:
    out = []
    for budget in GRID_BUDGETS:
        for ratio in GRID_RATIOS + GRID_EXTRA_RATIOS.get(budget, ()):
            s = size_for_budget(budget, ratio)
            lr = lr_for(s.n_params, rule)
            out.append(make_config("grid", f"grid/{budget:.0e}/r{ratio:g}", budget, ratio, lr))
            if budget == GRID_BUDGETS[0]:
                out.append(
                    make_config("grid", f"grid/{budget:.0e}/r{ratio:g}/seed1", budget, ratio, lr, seed=1, data_offset=SEED1_OFFSET)
                )
    return out


def dynamics_base(rule) -> TrainConfig:
    s = size_for_budget(DYNAMICS_BUDGET, DYNAMICS_RATIO)
    return make_config("dynamics", "dyn/baseline", DYNAMICS_BUDGET, DYNAMICS_RATIO, lr_for(s.n_params, rule))


def dynamics_configs(rule) -> list[TrainConfig]:
    base = dynamics_base(rule)
    lr = base.peak_lr
    out = [base]
    for mult in (3, 10):
        out.append(replace(base, name=f"dyn/lr{mult}x", peak_lr=lr * mult))
        out.append(replace(base, name=f"dyn/lr{mult}x-noclip", peak_lr=lr * mult, grad_clip=None))
    out += [
        replace(base, name="dyn/warmup0", warmup_frac=0.0),
        replace(base, name="dyn/warmup1", warmup_frac=0.01),
        replace(base, name="dyn/beta2-0.999", beta2=0.999),
    ]
    for bt in (32768, 131072, 262144, 524288):  # same LR at every batch size: noted as a limitation
        out.append(make_config("dynamics", f"dyn/batch{bt // 1024}k", DYNAMICS_BUDGET, DYNAMICS_RATIO, lr, batch_tokens=bt))
    return out


def dynamics_fix_configs(rule, from_run: str) -> list[TrainConfig]:
    src = {c.name: c for c in dynamics_configs(rule)}[from_run]
    return [replace(src, name=f"{from_run}+qknorm", qk_norm=True), replace(src, name=f"{from_run}+zloss", z_loss=1e-4)]


def declared_test_config(n_opt: float, rule, budget: float = TEST_BUDGET) -> TrainConfig:
    shape = choose_shape(n_opt)
    n = param_count(shape.n_layer, shape.d_model)
    steps = max(1, round(budget / (6 * n * BATCH_TOKENS)))
    return TrainConfig(
        name=f"test/{budget:.0e}", phase="test", n_layer=shape.n_layer, d_model=shape.d_model, total_steps=steps,
        peak_lr=lr_for(n, rule), budget_flops=budget, tokens_per_param=steps * BATCH_TOKENS / n,
    )


def configs_for_phase(phase, lr_rule="lr_rule.json", from_run="", prediction="predictions/3e17.json") -> list[TrainConfig]:
    if phase == "smoke":
        return smoke_configs()
    if phase == "lr":
        return lr_configs()
    if phase == "test":
        return [TrainConfig(**json.loads(Path(prediction).read_text())["declared_config"])]
    rule = load_lr_rule(lr_rule)
    if phase == "grid":
        return grid_configs(rule)
    if phase == "dynamics":
        return dynamics_configs(rule)
    if phase == "dynamics-fix":
        return dynamics_fix_configs(rule, from_run)
    raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")


def bench_shapes() -> list[tuple[int, int]]:
    """Smallest, largest and two middle grid shapes, for the throughput benchmark."""
    sizes = sorted({(s.n_params, s.n_layer, s.d_model) for b in GRID_BUDGETS for r in GRID_RATIOS for s in [size_for_budget(b, r)]})
    picks = [sizes[0], sizes[len(sizes) // 3], sizes[2 * len(sizes) // 3], sizes[-1]]
    return [(L, d) for _, L, d in picks]


def throughput_model(bench_path="bench.json"):
    """flops/s as a function of N: log-log interpolation of bench.json, else a constant."""
    p = Path(bench_path)
    if not p.exists():
        return lambda n: DEFAULT_FLOPS_PER_S
    rows = sorted(json.loads(p.read_text()), key=lambda r: r["n_params"])
    xs = np.log([r["n_params"] for r in rows])
    ys = np.log([r["flops_per_s"] for r in rows])
    return lambda n: float(np.exp(np.interp(math.log(n), xs, ys)))


def done_run_ids(results_dir="results") -> set[str]:
    d = Path(results_dir)
    ids = {json.loads(p.read_text())["run_id"] for p in d.glob("*.json")} if d.exists() else set()
    if (d / "failed").exists():
        ids |= {json.loads(p.read_text())["run_id"] for p in (d / "failed").glob("*.json")}
    return ids


def unique(cfgs: list[TrainConfig]) -> list[TrainConfig]:
    seen, out = set(), []
    for c in cfgs:
        if run_id(c) not in seen:
            seen.add(run_id(c))
            out.append(c)
    return out


def print_plan(cfgs: list[TrainConfig], done: set[str], bench_path="bench.json") -> None:
    tput = throughput_model(bench_path)
    cfgs = unique(cfgs)
    total_h = 0.0
    print(f"{'name':40} {'run_id':12} {'L':>3} {'d':>5} {'N':>10} {'tokens':>12} {'lr':>9} {'hours':>6} status")
    for c in cfgs:
        n = param_count(c.n_layer, c.d_model, c.vocab_size, c.qk_norm)
        tokens = c.total_steps * c.batch_tokens
        hours = (6 * n * tokens / tput(n) + RUN_OVERHEAD_S) / 3600
        status = "skip" if run_id(c) in done else "run"
        if status == "run":
            total_h += hours
        print(f"{c.name:40} {run_id(c):12} {c.n_layer:3d} {c.d_model:5d} {n:10,d} {tokens:12,d} {c.peak_lr:9.3g} {hours:6.2f} {status}")
    n_run = sum(run_id(c) not in done for c in cfgs)
    print(f"{len(cfgs)} runs ({n_run} to run): est. {total_h:.2f} GPU-hours, ${total_h * H100_DOLLARS_PER_HOUR:.2f}")


def _load_results(results_dir) -> list[dict]:
    return [json.loads(p.read_text()) for p in Path(results_dir).glob("*.json")]


def best_lrs(cfgs: list[TrainConfig], results: list[dict]) -> list[dict]:
    """Per LR-sweep group: losses aligned with LRs (missing/failed = +inf), best LR, interior flag."""
    by_id = {r["run_id"]: r for r in results}
    groups: dict[tuple[float, float], list[TrainConfig]] = {}
    for c in cfgs:
        groups.setdefault((c.budget_flops, c.tokens_per_param), []).append(c)
    out = []
    for (budget, ratio), members in groups.items():
        members.sort(key=lambda c: c.peak_lr)
        losses = [by_id[run_id(c)]["val_loss"] if run_id(c) in by_id else math.inf for c in members]
        i = int(np.argmin(losses))
        out.append({
            "budget": budget, "ratio": ratio, "n_params": param_count(members[0].n_layer, members[0].d_model),
            "lrs": [c.peak_lr for c in members], "losses": losses, "best_lr": members[i].peak_lr,
            "interior": 0 < i < len(members) - 1,
        })
    order = {(b, r): k for k, (b, r, _, _) in enumerate(LR_SWEEP)}
    return sorted(out, key=lambda g: order[(g["budget"], g["ratio"])])


def fit_lr_rule(groups: list[dict]) -> dict:
    """Line in (ln N, ln LR*) through the two sweep ends; the third group is the check."""
    (small, large, check) = groups[:3]
    x = np.log([small["n_params"], large["n_params"]])
    y = np.log([small["best_lr"], large["best_lr"]])
    c1, c0 = np.polyfit(x, y, 1)
    predicted = math.exp(c0 + c1 * math.log(check["n_params"]))
    return {"c0": float(c0), "c1": float(c1), "check_ratio": predicted / check["best_lr"], "groups": groups}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("phase", choices=PHASES)
    p.add_argument("--lr-rule", default="lr_rule.json")
    p.add_argument("--from-run", default="")
    p.add_argument("--prediction", default="predictions/3e17.json")
    p.add_argument("--results", default="results")
    p.add_argument("--bench", default="bench.json")
    f = sub.add_parser("fit-lr-rule")
    f.add_argument("--results", default="results")
    f.add_argument("--out", default="lr_rule.json")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        cfgs = configs_for_phase(args.phase, args.lr_rule, args.from_run, args.prediction)
        print_plan(cfgs, done_run_ids(args.results), args.bench)
    else:
        groups = best_lrs(lr_configs(), _load_results(args.results))
        for g in groups:
            flag = "ok" if g["interior"] else "EDGE: extend LR_SWEEP centre toward this end and rerun"
            print(f"{g['budget']:.0e} r{g['ratio']:g} N={g['n_params']:,} best_lr={g['best_lr']:.3g} {flag}")
        rule = fit_lr_rule(groups)
        print(f"rule: ln lr = {rule['c0']:.4f} + {rule['c1']:.4f} ln N; check ratio {rule['check_ratio']:.2f} (want 0.5-2)")
        Path(args.out).write_text(json.dumps(rule, indent=2))


if __name__ == "__main__":
    main()
