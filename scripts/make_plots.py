"""Build results/summary.csv and the three core figures from results/*/metrics.json.

1. Sample efficiency: accuracy at the training length vs. N, one line per k.
2. Length generalization: accuracy vs. test length, one line per k.
3. Attention: best head's edge-attention share vs. step index, one line per k
   (needs results/<run_id>/attention.json; make it with `python -m scratchpad.attention --run ...`).

Lines are the mean over seeds; the shaded band is the min-max range over seeds.
Run folders starting with "_" inside the results folder are ignored; point --results-dir at one
of them (for example results/_cluster_16nodes_6step_fresh) to plot it. A figure with no data is skipped.

    python scripts/make_plots.py                 # figures go to results/figures/
    python scripts/make_plots.py --n-train 16000 # N used for figures 2 and 3
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Colour follows the condition, not its rank: the same k keeps its colour in every figure.
COLORS = {"1": "#2a78d6", "2": "#1baf7a", "4": "#eda100", "full": "#eb6834"}
EXTRA_COLORS = ["#e87ba4", "#7a5cc7"]  # for other k values, should any be added
LABELS = {"1": "latest state only (k=1)", "2": "last 2 states", "4": "last 4 states", "full": "full history"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_runs(results_dir: Path) -> list[dict]:
    """One dict per finished run: k, n_train, seed, steps, eval (per test length), attention (or None)."""
    runs = []
    for metrics_path in sorted(results_dir.glob("[!_]*/metrics.json")):
        run_dir = metrics_path.parent
        cfg = json.loads((run_dir / "config.json").read_text())
        metrics = json.loads(metrics_path.read_text())
        k = cfg["train"]["history_k"]
        att_path = run_dir / "attention.json"
        runs.append(
            {
                "run_id": run_dir.name,
                "k": "full" if k is None else str(k),
                "n_train": cfg["train"]["n_train_instances"],
                "fresh": cfg["train"].get("fresh_data", False),
                "seed": cfg["train"]["seed"],
                "train_steps": cfg["task"]["train_steps"],
                "steps": metrics["steps"],
                "examples_seen": metrics["examples_seen"],
                "eval": metrics["eval"],
                "attention": json.loads(att_path.read_text()) if att_path.exists() else None,
            }
        )
    return runs


def write_summary(runs: list[dict], path: Path) -> None:
    """One row per run and test length."""
    fields = ["run_id", "k", "n_train", "seed", "optimizer_steps", "examples_seen", "test_length",
              "final_accuracy", "trace_accuracy", "format_failure_rate"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in runs:
            for length, m in r["eval"].items():
                w.writerow([r["run_id"], r["k"], r["n_train"], r["seed"], r["steps"], r["examples_seen"],
                            length, m["final_accuracy"], m["trace_accuracy"], m["format_failure_rate"]])


def aggregate(points: list[tuple[str, float, float]]) -> dict[str, list[tuple[float, float, float, float]]]:
    """(k, x, y) points -> k -> sorted [(x, mean, min, max)] over seeds."""
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for k, x, y in points:
        grouped[k][x].append(y)
    return {
        k: [(x, sum(ys) / len(ys), min(ys), max(ys)) for x, ys in sorted(xs.items())]
        for k, xs in grouped.items()
    }


def color_of(k: str, order: list[str]) -> str:
    if k in COLORS:
        return COLORS[k]
    others = [o for o in order if o not in COLORS]
    return EXTRA_COLORS[others.index(k) % len(EXTRA_COLORS)]


def line_figure(series, title, xlabel, ylabel, path, *, xticks=None, logx=False, chance=None) -> bool:
    """One line per k with a min-max band. Returns False (and writes nothing) if there is no data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not series:
        return False
    order = sorted(series, key=lambda k: (k == "full", int(k) if k.isdigit() else 0))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # With more than two conditions, lines that coincide would hide each other, so earlier lines are
    # drawn thicker and each later one sits visibly inside it.
    widen = len(order) > 2
    for i, k in enumerate(order):
        xs, means, los, his = zip(*series[k])
        color = color_of(k, order)
        extra = (len(order) - 1 - i) if widen else 0
        ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)
        ax.plot(xs, means, color=color, linewidth=2 + 2 * extra, marker="o", markersize=7 + 3 * extra,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS.get(k, f"k={k}"), clip_on=False)
    if chance is not None:
        ax.axhline(chance, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"chance {chance:.3g}", xy=(1, chance), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    if logx:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks, [f"{t:,}" if isinstance(t, int) else str(t) for t in xticks])
        ax.minorticks_off()
    ax.set_ylim(-0.02, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ncols = len(order) if len(order) <= 2 else 2
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=30 + 16 * (-(-len(order) // ncols) - 1))
    # Above the plot, so it can never cover a line.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=ncols, fontsize=10,
              labelcolor=INK, borderaxespad=0.2, handlelength=1.8, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results", type=Path)
    p.add_argument("--n-train", type=int, default=16000, help="training size for figures 2 and 3")
    p.add_argument("--metric", default="trace_accuracy", choices=["trace_accuracy", "final_accuracy"])
    args = p.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no finished runs in {args.results_dir}")
    out = args.results_dir / "figures"
    out.mkdir(exist_ok=True)
    write_summary(runs, args.results_dir / "summary.csv")
    print(f"{len(runs)} runs -> {args.results_dir / 'summary.csv'}")
    metric_name = args.metric.replace("_", " ")
    at_n = [r for r in runs if r["n_train"] == args.n_train]
    n_seeds = len({r["seed"] for r in at_n})
    # Fresh-data runs draw new problems at every step, so they have no training-set size to quote.
    data_text = "fresh training data" if at_n and all(r["fresh"] for r in at_n) else f"N={args.n_train:,}"

    # 1. Sample efficiency, at the training length and at the longest test length.
    sizes = sorted({r["n_train"] for r in runs})
    for which in ("train", "longest"):
        points = []
        for r in runs:
            length = str(r["train_steps"]) if which == "train" else max(r["eval"], key=int)
            points.append((r["k"], r["n_train"], r["eval"][length][args.metric]))
        length_text = f"{length}-step problems" + (" (training length)" if which == "train" else " (longest)")
        path = out / f"sample_efficiency_{which}.png"
        if len(sizes) > 1 and line_figure(
            aggregate(points), f"Held-out {metric_name} on {length_text}", "training instances N (log scale)",
            metric_name, path, xticks=sizes, logx=True,
        ):
            print(f"wrote {path}")

    # 2. Length generalization.
    points = [(r["k"], int(length), m[args.metric]) for r in at_n for length, m in r["eval"].items()]
    path = out / "length_generalization.png"
    lengths = sorted({x for _, x, _ in points})
    if line_figure(
        aggregate(points),
        f"Held-out {metric_name} by problem length ({data_text}, {n_seeds} seeds)" if not at_n or not at_n[0]["fresh"]
        else f"Held-out {metric_name} by problem length (fresh data, {n_seeds} seeds)",
        f"steps needed (trained on {at_n[0]['train_steps']} only)" if at_n else "", metric_name, path, xticks=lengths,
    ):
        print(f"wrote {path}")

    # 3. Attention: the run's best head, on the longest test length, by step index.
    points, chance = [], None
    for r in at_n:
        if r["attention"] is None:
            continue
        report = r["attention"][max(r["attention"], key=int)]
        chance = report["chance"]
        points += [(r["k"], int(step) + 1, share) for step, share in report["best_head_by_step"].items()]
    path = out / "attention_share.png"
    steps = sorted({x for _, x, _ in points})
    if line_figure(
        aggregate(points), f"Attention on the correct edge, best head ({data_text})",
        "step of the walk", "share of edge attention", path, xticks=steps, chance=chance,
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
