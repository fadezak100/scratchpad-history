"""Milestone 2 exit check: can the model memorize a small training set?

Trains on a few instances and then solves those SAME instances step by step with
evaluate.solve_batch. Passing means the data, loss mask, model, and solving loop all agree.
It says nothing about generalization.

    python scripts/smoke_overfit.py              # k = 1 and full history, 100 instances
    python scripts/smoke_overfit.py --k 1 --n 100 --max-steps 3000
"""
from __future__ import annotations

import argparse
import random

from scratchpad.config import ExperimentConfig
from scratchpad.evaluate import evaluate
from scratchpad.task import make_dataset
from scratchpad.train import run_training


def overfit(k: int | None, n_instances: int, max_steps: int, seed: int, device: str) -> dict:
    cfg = ExperimentConfig()
    cfg.train.history_k = k
    cfg.train.n_train_instances = n_instances
    cfg.train.max_steps = max_steps
    cfg.train.eval_every = 250
    cfg.train.early_stop_patience = 2
    cfg.train.seed = seed
    cfg.train.device = device
    instances = make_dataset(n_instances, cfg.task.n_nodes, cfg.task.train_steps, random.Random(seed))
    # The training set doubles as the "validation" set here: we are testing memorization.
    model, log = run_training(cfg, instances, instances)
    result = evaluate(model, {cfg.task.train_steps: instances}, k)[cfg.task.train_steps]
    return {"k": "full" if k is None else k, "steps": log["steps"], "seconds": log["train_seconds"],
            "n_params": log["n_params"], **result}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=None, help="history length; 0 = full. Default: run 1 and full")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    ks = [1, None] if args.k is None else [None if args.k == 0 else args.k]
    results = [overfit(k, args.n, args.max_steps, args.seed, args.device) for k in ks]
    print()
    for r in results:
        print(
            f"k={r['k']!s:>4}  params={r['n_params']:,}  steps={r['steps']}  time={r['seconds']:.0f}s  "
            f"final={r['final_accuracy']:.3f}  trace={r['trace_accuracy']:.3f}  "
            f"format_failures={r['format_failure_rate']:.3f}"
        )
    if not all(r["trace_accuracy"] >= 0.99 for r in results):
        raise SystemExit("FAILED: the model did not memorize the training set")
    print("PASSED")


if __name__ == "__main__":
    main()
