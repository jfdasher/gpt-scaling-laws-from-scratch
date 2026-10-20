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
