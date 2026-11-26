"""Training loop and CLI.

    python -m scratchpad.train --k 1 --n-train 16000 --seed 0   # --k 0 = full history
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, TrainConfig
from .data import ExampleDataset, FreshExampleStream, collate, keys_of, make_eval_sets, make_val_set
from .evaluate import evaluate
from .model import build_model, count_parameters
from .task import Instance, make_dataset
from .tokens import VOCAB
from .utils import pick_device, run_id, save_json, set_seed


def make_train_instances(cfg: ExperimentConfig) -> list[Instance]:
    """Training graphs depend on the seed and N only, never on k (invariant 2)."""
    rng = random.Random(f"train-{cfg.train.seed}")
    return make_dataset(cfg.train.n_train_instances, cfg.task.n_nodes, cfg.task.train_steps, rng)


def lr_at(step: int, tc: TrainConfig) -> float:
    """Learning rate for optimizer step `step` (0-based)."""
    if tc.lr_schedule == "constant":
        return tc.lr
    if tc.lr_schedule == "cosine":
        progress = min(step / max(1, tc.max_steps), 1.0)
        return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr_schedule {tc.lr_schedule!r}")


def build_optimizer(model: torch.nn.Module, tc: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay on weight matrices only (1-D parameters such as norm gains are exempt)."""
    matrices = [p for p in model.parameters() if p.dim() >= 2]
    vectors = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": matrices, "weight_decay": tc.weight_decay},
        {"params": vectors, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(0.9, tc.beta2))


def run_training(
    cfg: ExperimentConfig,
    train_instances: list[Instance],
    val_instances: list[Instance],
    probe_sets: dict[int, list[Instance]] | None = None,
    fresh_exclude: set | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Train on train_instances; every eval_every steps solve val_instances and maybe stop early.

    With cfg.train.fresh_data, train_instances is ignored: every step draws new instances, skipping
    any whose key is in fresh_exclude (the held-out keys, invariant 6).

    probe_sets (longer problems) are solved at the same moments and only logged. They never
    influence stopping, so they show how length generalization develops without selecting on it.
    Returns the model and a log with the loss/validation history and the amount of data seen.
    """
    tc = cfg.train
    set_seed(tc.seed)  # fixes the model's random initial weights
    device = pick_device(tc.device)
    model = build_model(cfg.model, len(VOCAB)).to(device)
    n_params = count_parameters(model)
    optimizer = build_optimizer(model, tc)

    if tc.fresh_data:
        stream = FreshExampleStream(
            cfg.task.n_nodes, cfg.task.train_steps, tc.history_k, tc.seed, fresh_exclude or set(),
            tc.true_positions,
        )
        loader = DataLoader(stream, batch_size=tc.batch_size, collate_fn=collate)
    else:
        dataset = ExampleDataset(train_instances, tc.history_k, tc.true_positions)
        loader = DataLoader(
            dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(tc.seed),
        )
    val_sets = {cfg.task.train_steps: val_instances}

    history: list[dict] = []
    step = examples_seen = tokens_seen = 0
    best_val, evals_since_best, stopped_early, out_of_time = -1.0, 0, False, False
    recent_losses: list[float] = []
    started = time.time()
    model.train()
    while step < tc.max_steps and not stopped_early and not out_of_time:
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, tc)
            loss = model(**batch).loss
            loss.backward()
            if tc.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            examples_seen += batch["input_ids"].shape[0]
            tokens_seen += int(batch["attention_mask"].sum())
            recent_losses.append(loss.item())

            if step % tc.eval_every == 0 or step == tc.max_steps:
                val = evaluate(model, val_sets, tc.history_k, tc.true_positions)[cfg.task.train_steps]
                history.append(
                    {
                        "step": step,
                        "train_loss": sum(recent_losses) / len(recent_losses),
                        "val_final_accuracy": val["final_accuracy"],
                        "val_trace_accuracy": val["trace_accuracy"],
                        "examples_seen": examples_seen,
                        "tokens_seen": tokens_seen,
                    }
                )
                line = (
                    f"step {step:>6}  loss {history[-1]['train_loss']:.4f}  "
                    f"val final {val['final_accuracy']:.3f}  val trace {val['trace_accuracy']:.3f}"
                )
                if probe_sets and val["trace_accuracy"] > 0:  # pointless before 6 steps work at all
                    probe = evaluate(model, probe_sets, tc.history_k, tc.true_positions)
                    history[-1]["probe_final_accuracy"] = {
                        str(n): m["final_accuracy"] for n, m in probe.items()
                    }
                    line += "  probe " + " ".join(
                        f"{n}:{m['final_accuracy']:.2f}" for n, m in probe.items()
                    )
                print(line, flush=True)
                recent_losses = []
                if val["final_accuracy"] > best_val:
                    best_val, evals_since_best = val["final_accuracy"], 0
                elif best_val >= tc.early_stop_min_accuracy:
                    evals_since_best += 1
                    stopped_early = evals_since_best >= tc.early_stop_patience
            out_of_time = tc.time_limit_minutes > 0 and time.time() - started > 60 * tc.time_limit_minutes
            if step >= tc.max_steps or stopped_early or out_of_time:
                break

    log = {
        "n_params": n_params,
        "device": device,
        "steps": step,
        "stopped_early": stopped_early,
        "stopped_by_time_limit": out_of_time,
        "examples_seen": examples_seen,
        "tokens_seen": tokens_seen,
        # Rough rule of thumb: ~2 FLOPs per parameter per token forward, ~3x that with backward.
        "est_train_flops": 6 * n_params * tokens_seen,
        "train_seconds": time.time() - started,
        "history": history,
    }
    return model, log


def train(cfg: ExperimentConfig) -> dict:
    """Train one run; evaluate every cfg.train.eval_every steps; early-stop on 6-step val accuracy.

    Writes results/<run_id>/config.json and metrics.json. Logs examples, tokens, and
    estimated FLOPs seen, so runs can be compared per example and per token.
    """
    tc = cfg.train
    out_dir = Path(cfg.results_dir) / run_id(tc.history_k, tc.n_train_instances, tc.seed, tc.true_positions)
    save_json(cfg, out_dir / "config.json")

    train_instances = [] if tc.fresh_data else make_train_instances(cfg)
    train_keys = keys_of(train_instances)
    eval_sets = make_eval_sets(cfg.task, exclude=train_keys)
    eval_keys = {key for instances in eval_sets.values() for key in keys_of(instances)}
    val_instances = make_val_set(cfg.task, exclude=train_keys | eval_keys)

    probe_sets = {
        n: instances[: tc.n_probe_per_length]
        for n, instances in eval_sets.items()
        if n != cfg.task.train_steps and tc.n_probe_per_length > 0
    }

    model, log = run_training(
        cfg, train_instances, val_instances, probe_sets, fresh_exclude=eval_keys | keys_of(val_instances)
    )
    final = evaluate(model, eval_sets, tc.history_k, tc.true_positions)
    metrics = {**log, "eval": {str(n): m for n, m in final.items()}}
    torch.save(model.state_dict(), out_dir / "model.pt")  # for the attention analysis; git-ignored
    save_json(metrics, out_dir / "metrics.json")  # written last: its presence marks the run as finished
    return metrics


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=0, help="history length; 0 = full, 1 = inductive")
    p.add_argument("--n-train", type=int, default=16000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None, help="default: TrainConfig.max_steps")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-nodes", type=int, default=None, help="default: TaskConfig.n_nodes")
    p.add_argument("--train-steps", type=int, default=None, help="default: TaskConfig.train_steps")
    p.add_argument("--test-steps", default=None, help="comma-separated test lengths, e.g. 6,8,10,12")
    p.add_argument("--fresh", action="store_true",
                   help="pilot: new training instances every step (ignores --n-train)")
    p.add_argument("--eval-every", type=int, default=None, help="default: TrainConfig.eval_every")
    p.add_argument("--results-dir", default=None, help="default: results")
    p.add_argument("--true-positions", action="store_true",
                   help="position-reset control: hide history but keep tokens at their full-sequence positions")
    p.add_argument("--time-limit-min", type=float, default=None,
                   help="stop training after this many minutes, then evaluate and save")
    p.add_argument("--n-layers", type=int, default=None, help="default: ModelConfig.num_layers")
    p.add_argument("--n-heads", type=int, default=None, help="default: ModelConfig.num_heads")
    p.add_argument("--hidden-size", type=int, default=None, help="model width. default: ModelConfig.hidden_size")
    p.add_argument("--ffn-size", type=int, default=None,
                   help="feed-forward width. default: ModelConfig.intermediate_size")
    args = p.parse_args(argv)
    cfg = ExperimentConfig()
    cfg.train.history_k = None if args.k == 0 else args.k
    cfg.train.n_train_instances = args.n_train
    cfg.train.seed = args.seed
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    cfg.train.device = args.device
    if args.n_nodes is not None:
        cfg.task.n_nodes = args.n_nodes
    if args.train_steps is not None:
        cfg.task.train_steps = args.train_steps
    if args.test_steps is not None:
        cfg.task.test_steps = tuple(int(part) for part in args.test_steps.split(","))
    cfg.train.fresh_data = args.fresh
    if args.eval_every is not None:
        cfg.train.eval_every = args.eval_every
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    cfg.train.true_positions = args.true_positions
    if args.time_limit_min is not None:
        cfg.train.time_limit_minutes = args.time_limit_min
    if args.n_layers is not None:
        cfg.model.num_layers = args.n_layers
    if args.n_heads is not None:
        cfg.model.num_heads = args.n_heads
    if args.hidden_size is not None:
        cfg.model.hidden_size = args.hidden_size
    if args.ffn_size is not None:
        cfg.model.intermediate_size = args.ffn_size
    return cfg


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
