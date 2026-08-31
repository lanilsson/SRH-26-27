"""Aggregate results/runs/*.json into the summary table for Data Analysis.

Produces results/tables/summary.csv with mean +/- SEM over the 5 seeds for
each dataset x condition, for all three dependent variables.

Usage: python scripts/aggregate.py
"""

import json
from pathlib import Path

import pandas as pd

RUNS_DIR = Path("results/runs")
OUT = Path("results/tables/summary.csv")

METRICS = ["pr_auc", "f1", "roc_auc"]


def main():
    records = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        run = json.loads(path.read_text())
        records.append({
            "dataset": run["dataset"],
            "condition": run["condition"],
            "seed": run["seed"],
            **{m: run["metrics"][m] for m in METRICS},
        })
    if not records:
        raise SystemExit(f"No run files found in {RUNS_DIR}/")

    df = pd.DataFrame(records)
    grouped = df.groupby(["dataset", "condition"])
    summary = grouped[METRICS].agg(["mean", "sem"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_runs"] = grouped.size()
    summary = summary.round(4).reset_index()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT} ({len(df)} runs aggregated)")


if __name__ == "__main__":
    main()
