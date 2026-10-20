"""Loading pulled results (written by `modal run modal_app.py::pull_results`)."""

from __future__ import annotations

import json
from pathlib import Path


def load_results(results_dir="results") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]


def load_failed(results_dir="results") -> list[dict]:
    d = Path(results_dir) / "failed"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def load_curve(results_dir, run_id: str) -> list[dict]:
    """Per-step rows; after a preemption the later copy of a step wins."""
    rows: dict[int, dict] = {}
    for line in (Path(results_dir) / "curves" / f"{run_id}.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[r["step"]] = r
    return [rows[s] for s in sorted(rows)]


def load_diag(results_dir, run_id: str) -> list[dict]:
    """Per-layer diagnostic rows (grad norms, attention max-logits) by step; empty if the run logged none."""
    path = Path(results_dir) / "curves" / f"{run_id}.diag.jsonl"
    if not path.exists():
        return []
    rows: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[r["step"]] = r
    return [rows[s] for s in sorted(rows)]


def grid_run_ids(version: str = "v2", lr_rule: str = "lr_rule.json") -> set[str]:
    """Run ids of the grid as defined by code (sweep.py), so analysis does not depend on the `phase` label alone."""
    import sweep
    from gptlab.train import run_id

    rule = sweep.load_lr_rule(lr_rule)
    cfgs = sweep.grid_v1_configs(rule) if version == "v1" else sweep.grid_configs(rule)
    return {run_id(c) for c in cfgs}
