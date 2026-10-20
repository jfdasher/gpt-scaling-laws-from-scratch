import json
import math
import sys

import pytest

import sweep
from gptlab.train import run_id

RULE = (math.log(3e-3) + 0.3 * math.log(2.5e6), -0.3)  # lr = 3e-3 at 2.5M, falling as N^-0.3


def test_grid_configs():
    cfgs = sweep.grid_configs(RULE)
    assert len(cfgs) == 25
    assert len({run_id(c) for c in cfgs}) == 25
    seed1 = [c for c in cfgs if c.seed == 1]
    assert len(seed1) == 5
    assert all(c.data_offset == sweep.SEED1_OFFSET and c.budget_flops == 3e15 for c in seed1)


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
    grid = {c.name: c for c in sweep.grid_configs(RULE)}
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
