"""Cost ledger from result.json (and failed runs' forensic GPU time).

    uv run python -m analysis.costs results
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from analysis.common import load_failed, load_results
from gptlab.flops import H100_DOLLARS_PER_HOUR


def cost_line(r: dict) -> str:
    return f"this run: {r['gpu_hours']:.2f} GPU-hours, ${r['cost_usd']:.2f}"


def _rows(results, failed, dollars_per_hour):
    for r in results:
        yield {"run_id": r["run_id"], "name": r["name"], "phase": r["phase"], "status": "ok",
               "gpu_hours": r["gpu_hours"], "cost_usd": r["cost_usd"]}
    for f in failed:
        hours = f.get("forensic", {}).get("gpu_seconds", 0.0) / 3600
        yield {"run_id": f["run_id"], "name": f["name"], "phase": f["phase"], "status": "failed",
               "gpu_hours": hours, "cost_usd": hours * dollars_per_hour}


def phase_totals(results, failed, dollars_per_hour=H100_DOLLARS_PER_HOUR) -> dict:
    tot: dict = defaultdict(lambda: {"runs": 0, "gpu_hours": 0.0, "cost_usd": 0.0})
    for row in _rows(results, failed, dollars_per_hour):
        t = tot[row["phase"]]
        t["runs"] += 1
        t["gpu_hours"] += row["gpu_hours"]
        t["cost_usd"] += row["cost_usd"]
    return dict(tot)


def write_costs_csv(results, failed, path, dollars_per_hour=H100_DOLLARS_PER_HOUR) -> None:
    rows = list(_rows(results, failed, dollars_per_hour))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "name", "phase", "status", "gpu_hours", "cost_usd"])
        w.writeheader()
        w.writerows(rows)


def main(results_dir="results") -> None:
    res, failed = load_results(results_dir), load_failed(results_dir)
    write_costs_csv(res, failed, Path(results_dir) / "costs.csv")
    totals = phase_totals(res, failed)
    for phase, t in sorted(totals.items()):
        print(f"{phase:14} {t['runs']:3d} runs {t['gpu_hours']:7.2f} GPU-h ${t['cost_usd']:8.2f}")
    print(f"{'total':14} {sum(t['runs'] for t in totals.values()):3d} runs "
          f"{sum(t['gpu_hours'] for t in totals.values()):7.2f} GPU-h ${sum(t['cost_usd'] for t in totals.values()):8.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
