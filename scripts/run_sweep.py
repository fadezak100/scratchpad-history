"""Run the plan.md §3 matrix: k x N x seed. Skips runs whose metrics.json exists (resumable).

    python scripts/run_sweep.py --dry-run
    python scripts/run_sweep.py --ks 1,full --sizes 16000 --seeds 0,1,2    # milestone 3
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

HISTORY_KS = [1, None]  # None = full. With 3-step training, k >= 2 already equals full history.
TRAIN_SIZES = [250, 1_000, 4_000, 16_000]  # 64k dropped: only ~80k distinct instances exist (plan.md §9)
SEEDS = [0, 1, 2]


def matrix(ks=HISTORY_KS, sizes=TRAIN_SIZES, seeds=SEEDS):
    return list(itertools.product(ks, sizes, seeds))


def parse_ks(text: str) -> list[int | None]:
    """'1,full' -> [1, None]."""
    return [None if part.strip() == "full" else int(part) for part in text.split(",")]


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",")]


def k_name(k: int | None) -> str:
    return "full" if k is None else str(k)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ks", type=parse_ks, default=HISTORY_KS, help="e.g. 1,2,4,full")
    p.add_argument("--sizes", type=parse_ints, default=TRAIN_SIZES, help="e.g. 1000,16000")
    p.add_argument("--seeds", type=parse_ints, default=SEEDS, help="e.g. 0,1,2")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    runs = matrix(args.ks, args.sizes, args.seeds)
    if args.dry_run:
        for k, n, seed in runs:
            print(f"k={k_name(k):>4}  N={n:>6}  seed={seed}")
        print(f"{len(runs)} runs")
        return

    # Imported here so --dry-run works without loading torch.
    from scratchpad.config import ExperimentConfig
    from scratchpad.train import train
    from scratchpad.utils import run_id

    for i, (k, n, seed) in enumerate(runs, start=1):
        name = run_id(k, n, seed)
        if (Path(args.results_dir) / name / "metrics.json").exists():
            print(f"[{i}/{len(runs)}] {name}: already done, skipping", flush=True)
            continue
        print(f"[{i}/{len(runs)}] {name}: starting", flush=True)
        cfg = ExperimentConfig(results_dir=args.results_dir)
        cfg.train.history_k = k
        cfg.train.n_train_instances = n
        cfg.train.seed = seed
        metrics = train(cfg)
        summary = "  ".join(f"{length}:{m['final_accuracy']:.3f}" for length, m in metrics["eval"].items())
        print(f"[{i}/{len(runs)}] {name}: done in {metrics['train_seconds']:.0f}s, "
              f"{metrics['steps']} steps, final accuracy by length  {summary}", flush=True)


if __name__ == "__main__":
    main()
