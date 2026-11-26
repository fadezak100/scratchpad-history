"""Training cost to reach a validation-accuracy threshold, one row per run.

For every run in a results folder, find the first evaluation whose validation trace accuracy
(share of validation problems with every state correct) is at least the threshold, and report the
optimizer steps, training examples and training tokens processed by then. Runs that never reach
it get empty cells. Evaluations happen every eval_every steps, so the values are rounded up.

    python scripts/threshold_table.py --results-dir results/_pilot_16nodes_6step_fresh
    python scripts/threshold_table.py --threshold 0.99 --csv results/thresholds.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scratchpad.utils import first_reaching

FIELDS = ["run_id", "k", "seed", "fresh_data", "n_train", "reached", "step", "examples_seen", "tokens_seen"]


def rows_for(results_dir: Path, threshold: float) -> list[dict]:
    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        cfg = json.loads((metrics_path.parent / "config.json").read_text())
        hit = first_reaching(json.loads(metrics_path.read_text())["history"], threshold)
        k = cfg["train"]["history_k"]
        rows.append(
            {
                "run_id": metrics_path.parent.name,
                "k": "full" if k is None else str(k),
                "seed": cfg["train"]["seed"],
                "fresh_data": cfg["train"].get("fresh_data", False),
                "n_train": cfg["train"]["n_train_instances"],
                "reached": hit is not None,
                "step": hit["step"] if hit else "",
                "examples_seen": hit["examples_seen"] if hit else "",
                "tokens_seen": hit["tokens_seen"] if hit else "",
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results", type=Path)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--csv", type=Path, help="also write the table to this file")
    args = p.parse_args()
    rows = rows_for(args.results_dir, args.threshold)
    print("  ".join(f"{f:>14}" for f in FIELDS))
    for r in rows:
        print("  ".join(f"{str(r[f]):>14}" for f in FIELDS))
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
