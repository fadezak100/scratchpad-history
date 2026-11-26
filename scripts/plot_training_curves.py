"""Validation trace accuracy during training, one panel per history condition, one line per seed.

    python scripts/plot_training_curves.py --results-dir results/exp_a

Reads the `history` in each run's metrics.json and writes <results-dir>/figures/training_curves.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"full": "#eb6834", "4": "#eda100", "2": "#1baf7a", "1": "#2a78d6"}
TITLES = {"full": "Full history", "4": "Last 4 states (k=4)", "2": "Last 2 states (k=2)",
          "1": "Latest state only (k=1)"}
LINE_STYLES = ["-", "--", ":"]


def load_histories(results_dir: Path) -> dict[str, dict[int, list[dict]]]:
    """{condition: {seed: history}} for every finished run in results_dir."""
    out: dict[str, dict[int, list[dict]]] = {}
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        cfg = json.loads((metrics_path.parent / "config.json").read_text())["train"]
        k = "full" if cfg["history_k"] is None else str(cfg["history_k"])
        out.setdefault(k, {})[cfg["seed"]] = json.loads(metrics_path.read_text())["history"]
    return out


def plot(histories: dict[str, dict[int, list[dict]]], threshold: float, out_path: Path) -> None:
    conditions = [k for k in TITLES if k in histories]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.3), sharex=True, sharey=True, squeeze=False)
    for ax, k in zip(axes.flat, conditions):
        for i, (seed, history) in enumerate(sorted(histories[k].items())):
            ax.plot([e["step"] / 1000 for e in history], [100 * e["val_trace_accuracy"] for e in history],
                    color=COLORS[k], lw=2, ls=LINE_STYLES[i % len(LINE_STYLES)], label=f"seed {seed}")
        ax.axhline(100 * threshold, color="#999999", lw=0.8)
        ax.set_title(TITLES[k], fontsize=10, loc="left")
        ax.set_ylim(-4, 104)
        ax.set_xlim(left=0)
        ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=9, length=0)
        ax.grid(axis="y", color="#e6e6e6", lw=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="center right")
    for ax in axes[1]:
        ax.set_xlabel("optimizer steps (thousands)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("validation trace acc. (%)", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.95)
    args = p.parse_args()
    results_dir = Path(args.results_dir)
    out_path = results_dir / "figures" / "training_curves.png"
    plot(load_histories(results_dir), args.threshold, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
