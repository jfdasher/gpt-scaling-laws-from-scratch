import json
import math
import sys

import pytest
from dataclasses import replace

import sweep
from gptlab.flops import size_for_budget
from gptlab.train import run_id

RULE = (math.log(3e-3) + 0.3 * math.log(2.5e6), -0.3)  # lr = 3e-3 at 2.5M, falling as N^-0.3


def test_grid_configs():
    cfgs = sweep.grid_configs(RULE)
    assert len(cfgs) == 25
    assert len({run_id(c) for c in cfgs}) == 25
    seed1 = [c for c in cfgs if c.seed == 1]
    assert len(seed1) == 5
    assert all(c.data_offset == sweep.SEED1_OFFSET and c.budget_flops == 3e15 for c in seed1)


def test_grid_v2_batch_per_budget():
    for c in sweep.grid_configs(RULE):
        assert c.batch_tokens == sweep.GRID_BATCH[c.budget_flops]
        assert c.batch_tokens % (c.micro_batch * c.seq_len) == 0
        assert c.micro_batch == min(64, c.batch_tokens // c.seq_len)
    assert all(c.batch_tokens == 65536 for c in sweep.grid_v1_configs(RULE))
    assert [c.name for c in sweep.grid_v1_configs(RULE)] == [c.name for c in sweep.grid_configs(RULE)]


def test_grid_v2_1e17_run_ids_equal_v1():
    v1 = {c.name: run_id(c) for c in sweep.grid_v1_configs(RULE) if c.budget_flops == 1e17}
    v2 = {c.name: run_id(c) for c in sweep.grid_configs(RULE) if c.budget_flops == 1e17}
    assert v1 and v1 == v2
    v1_small = {run_id(c) for c in sweep.grid_v1_configs(RULE) if c.budget_flops < 1e17}
    assert not v1_small & {run_id(c) for c in sweep.grid_configs(RULE)}


def test_reused_dyn_batch32k_is_in_grid_v2():
    by_name = {c.name: c for c in sweep.dynamics_configs(RULE)}
    assert run_id(by_name["dyn/batch32k"]) in {run_id(c) for c in sweep.grid_configs(RULE)}


def test_lr_configs():
    cfgs = sweep.lr_configs()
    assert len(cfgs) == 13
    by_group = {}
    for c in cfgs:
        by_group.setdefault((c.budget_flops, c.tokens_per_param), []).append(c.peak_lr)
    for lrs in by_group.values():
        lrs.sort()
        assert all(math.isclose(b / a, 2.0) for a, b in zip(lrs, lrs[1:]))


def test_dynamics_configs():
    cfgs = sweep.dynamics_configs(RULE)
    assert len(cfgs) == 12
    by_name = {c.name: c for c in cfgs}
    grid = {c.name: c for c in sweep.grid_v1_configs(RULE)}
    assert run_id(by_name["dyn/baseline"]) == run_id(grid["grid/3e+16/r20"])
    base_tokens = by_name["dyn/baseline"].total_steps * by_name["dyn/baseline"].batch_tokens
    for c in cfgs:
        if c.name.startswith("dyn/batch"):
            assert abs(c.total_steps * c.batch_tokens - base_tokens) <= c.batch_tokens
            assert c.batch_tokens % (c.micro_batch * c.seq_len) == 0
    assert by_name["dyn/lr10x-noclip"].grad_clip is None
    fixes = sweep.dynamics_fix_configs(RULE, "dyn/lr10x-noclip")
    assert [f.name for f in fixes] == ["dyn/lr10x-noclip+qknorm", "dyn/lr10x-noclip+zloss"]
    assert fixes[0].qk_norm and fixes[1].z_loss == 1e-4


def _result(cfg, loss):
    return {"run_id": run_id(cfg), "phase": "lr", "val_loss": loss, "n_params": 1, "config": {}}


def test_best_lrs_and_rule():
    cfgs = sweep.lr_configs()
    results = []
    for c in cfgs:
        if c.peak_lr == max(x.peak_lr for x in cfgs if x.budget_flops == c.budget_flops):
            continue  # highest LR "diverged": no result
        results.append(_result(c, abs(math.log(c.peak_lr / (c.peak_lr if c.tokens_per_param == 20 else 3e-3))) + 3))
    groups = sweep.best_lrs(cfgs, results)
    assert len(groups) == 3
    assert all(math.isinf(g["losses"][-1]) for g in groups)
    rule = sweep.fit_lr_rule(groups)
    assert {"c0", "c1", "check_ratio"} <= set(rule)


def test_declared_test_config():
    c = sweep.declared_test_config(5e7, RULE)
    from gptlab.flops import param_count

    n = param_count(c.n_layer, c.d_model)
    assert abs(6 * n * c.total_steps * c.batch_tokens / 3e17 - 1) < 0.01
    assert c.phase == "test"


def test_smoke_configs_distinct():
    a, b = sweep.smoke_configs()
    assert run_id(a) != run_id(b)


def test_plan_cli_does_not_import_modal(tmp_path, capsys):
    rule_path = tmp_path / "lr_rule.json"
    rule_path.write_text(json.dumps({"c0": RULE[0], "c1": RULE[1]}))
    sweep.main(["plan", "grid", "--lr-rule", str(rule_path), "--results", str(tmp_path / "results")])
    out = capsys.readouterr().out
    assert "25 runs" in out and "$" in out
    assert "modal" not in sys.modules


def test_grid_does_not_collide_with_lr_sweep():
    ends = [size_for_budget(b, r) for b, r, _, _ in sweep.LR_SWEEP[:2]]
    c1 = math.log(2e-3 / 4e-3) / math.log(ends[1].n_params / ends[0].n_params)
    c0 = math.log(4e-3) - c1 * math.log(ends[0].n_params)
    grid_ids = {run_id(c) for c in sweep.grid_configs((c0, c1))}
    assert not {run_id(c) for c in sweep.lr_configs()} & grid_ids
    # dyn/batch32k at 3e16 is, by design, the same run as grid v2's 3e16/r20 (32k batch); checked against v1
    v1_ids = {run_id(c) for c in sweep.grid_v1_configs((c0, c1))}
    for c in sweep.dynamics_configs((c0, c1)):
        if c.name != "dyn/baseline":
            assert run_id(c) not in v1_ids


def test_best_lrs_nan_not_selected():
    cfgs = sweep.lr_configs()[:5]
    losses = [3.0, 2.0, float("nan"), 2.5, 3.0]
    groups = sweep.best_lrs(cfgs, [_result(c, l) for c, l in zip(cfgs, losses)])
    assert groups[0]["best_lr"] == cfgs[1].peak_lr


def test_fit_lr_rule_cli_refuses_without_results(tmp_path):
    out = tmp_path / "lr_rule.json"
    args = ["fit-lr-rule", "--results", str(tmp_path / "none"), "--out", str(out)]
    with pytest.raises(SystemExit) as e:
        sweep.main(args)
    assert e.value.code == 1 and not out.exists()
    sweep.main(args + ["--force"])
    assert out.exists()


def test_diag_every_on_instability_configs_only():
    by_name = {c.name: c for c in sweep.dynamics_configs(RULE)}
    assert by_name["dyn/lr10x-noclip"].diag_every == 20
    assert all(c.diag_every == 20 for n, c in by_name.items() if n.startswith("dyn/lr"))
    assert by_name["dyn/baseline"].diag_every is None
    assert all(c.diag_every is None for c in sweep.grid_configs(RULE))
    grid = {c.name: c for c in sweep.grid_v1_configs(RULE)}
    assert run_id(by_name["dyn/baseline"]) == run_id(grid["grid/3e+16/r20"])
    assert run_id(by_name["dyn/lr10x"]) == run_id(replace(by_name["dyn/lr10x"], diag_every=None))
    assert all(f.diag_every == 20 for f in sweep.dynamics_fix_configs(RULE, "dyn/lr10x-noclip"))


def test_skip_set():
    remote = {"a": "done", "b": "failed", "c": "in_progress", "d": "in_progress"}
    skip, warns = sweep.skip_set({"x"}, remote, ["a", "b", "c", "d", "e", "x"])
    assert skip == {"x", "a", "b", "c", "d"}
    assert len(warns) == 2 and all("WARNING" in w for w in warns)
    skip, warns = sweep.skip_set(set(), remote, ["c", "e"], include_in_progress=True)
    assert skip == {"a", "b"} and len(warns) == 1
    skip, warns = sweep.skip_set({"c"}, remote, ["c"])  # locally done wins over a stale remote state
    assert "c" in skip and not warns
