"""Edge-attention share (plan.md §4).

At the position that predicts the next node, take attention weights to the context and compute
    attention on the correct edge's tokens / attention on all edge tokens.
Normalizing over edge tokens keeps shorter histories from inflating the score. Chance = 1/n_edges.

The next node is predicted at the "S" token that opens the target state, so the model input is
context + "S" and we read the attention row of that last position.

Contexts here hold the ORACLE's earlier states (teacher forcing), not the model's own. That is
fine for a measurement (nothing is scored as right or wrong), but on instances the model gets
wrong, the attention shown is for a context it would not have produced itself.

    python -m scratchpad.attention --run results/k1_n16000_s0     # writes attention.json there
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .formats import Example, make_examples
from .tokens import TOKEN_TO_ID


def share_from_attention(
    attn: torch.Tensor, spans: dict[int, tuple[int, int]], correct_node: int
) -> torch.Tensor:
    """attn has shape (..., seq_len): attention from one position to every position.

    Returns (...): weight on the edge leaving correct_node / weight on all edges.
    """
    lo, hi = spans[correct_node]
    on_correct = attn[..., lo:hi].sum(dim=-1)
    on_edges = sum(attn[..., a:b].sum(dim=-1) for a, b in spans.values())
    return on_correct / on_edges.clamp_min(1e-12)


@torch.no_grad()
def edge_attention_shares(model, examples: list[Example], batch_size: int = 256) -> torch.Tensor:
    """Tensor (len(examples), num_layers, num_heads), on the CPU. Needs eager attention."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    by_length: dict[int, list[int]] = defaultdict(list)  # equal lengths batch without padding
    for i, ex in enumerate(examples):
        by_length[len(ex.context)].append(i)
    out: list[torch.Tensor | None] = [None] * len(examples)
    for indices in by_length.values():
        for lo in range(0, len(indices), batch_size):
            chunk = indices[lo : lo + batch_size]
            ids = torch.tensor(
                [examples[i].context + [TOKEN_TO_ID["S"]] for i in chunk], dtype=torch.long, device=device
            )
            kwargs = {}
            if examples[chunk[0]].position_ids is not None:  # true-positions runs; +1 for the "S"
                kwargs["position_ids"] = torch.tensor(
                    [examples[i].position_ids[: ids.shape[1]] for i in chunk], dtype=torch.long, device=device
                )
            attentions = model(input_ids=ids, output_attentions=True, **kwargs).attentions
            # (batch, layers, heads, seq): the last position's attention row in every layer and head.
            last = torch.stack([a[:, :, -1, :] for a in attentions], dim=1).float().cpu()
            for row, i in enumerate(chunk):
                ex = examples[i]
                out[i] = share_from_attention(last[row], ex.edge_spans, ex.current_node)
    model.train(was_training)
    return torch.stack(out)  # type: ignore[arg-type]


def edge_attention_share(model, example: Example) -> torch.Tensor:
    """Tensor of shape (num_layers, num_heads) with the share for this example."""
    return edge_attention_shares(model, [example])[0]


def attention_report(model, examples: list[Example]) -> dict:
    """Mean share per layer/head, by step index, plus the best head (highest overall mean)."""
    shares = edge_attention_shares(model, examples)
    mean = shares.mean(dim=0)
    layer, head = divmod(int(mean.argmax()), mean.shape[1])
    steps = sorted({ex.step for ex in examples})
    by_step = {
        step: shares[[i for i, ex in enumerate(examples) if ex.step == step]].mean(dim=0) for step in steps
    }
    return {
        "n_examples": len(examples),
        "chance": 1 / len(examples[0].edge_spans),
        "mean": mean.tolist(),
        "best_head": {"layer": layer, "head": head, "share": float(mean[layer, head])},
        "by_step": {str(step): m.tolist() for step, m in by_step.items()},
        "best_head_by_step": {str(step): float(m[layer, head]) for step, m in by_step.items()},
    }


def report_for_run(run_dir: Path, n_per_length: int = 500) -> dict:
    """Load a finished run and report attention on its held-out sets, one report per test length."""
    from .config import ExperimentConfig, ModelConfig, TaskConfig, TrainConfig
    from .data import keys_of, make_eval_sets
    from .model import build_model
    from .tokens import VOCAB
    from .train import make_train_instances
    from .utils import pick_device

    raw = json.loads((run_dir / "config.json").read_text())
    raw["task"]["test_steps"] = tuple(raw["task"]["test_steps"])
    cfg = ExperimentConfig(
        task=TaskConfig(**raw["task"]), model=ModelConfig(**raw["model"]), train=TrainConfig(**raw["train"])
    )
    model = build_model(cfg.model, len(VOCAB))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.to(pick_device(cfg.train.device))
    # Same held-out sets as train(): a fresh-data run has no fixed training set to exclude.
    train_instances = [] if cfg.train.fresh_data else make_train_instances(cfg)
    eval_sets = make_eval_sets(cfg.task, exclude=keys_of(train_instances))
    k = cfg.train.history_k
    return {
        str(length): attention_report(
            model,
            [ex for inst in instances[:n_per_length] for ex in make_examples(inst, k, cfg.train.true_positions)],
        )
        for length, instances in eval_sets.items()
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="a results/<run_id>/ directory")
    p.add_argument("--n-per-length", type=int, default=500, help="held-out instances per test length")
    args = p.parse_args(argv)
    report = report_for_run(args.run, args.n_per_length)
    (args.run / "attention.json").write_text(json.dumps(report, indent=2))
    for length, r in report.items():
        b = r["best_head"]
        print(f"{length} steps: best head L{b['layer']}H{b['head']} share {b['share']:.3f} (chance {r['chance']:.3f})")


if __name__ == "__main__":
    main()
