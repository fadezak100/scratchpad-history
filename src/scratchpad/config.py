"""Dataclass configs. Single source of defaults; see plan.md §2-§3."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    # 8 nodes / 3-step training since 2026-11-21 (plan.md §9); the second paper uses 24 / 6 / up to 12.
    n_nodes: int = 8
    train_steps: int = 3
    test_steps: tuple[int, ...] = (3, 4, 5, 6)
    n_eval_per_length: int = 2000
    eval_seed: int = 12345  # fixed so every run sees the same held-out sets
    n_val: int = 500  # separate validation set at the training length, used only for early stopping
    val_seed: int = 54321


@dataclass
class ModelConfig:
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 4
    num_heads: int = 4
    max_positions: int = 2048
    # "eager" returns attention weights (needed for the attention analysis); "sdpa" is faster.
    # Both compute the same function, so weights trained with one load into the other.
    attn_implementation: str = "eager"


@dataclass
class TrainConfig:
    history_k: int | None = None  # None = full history; 1 = inductive
    n_train_instances: int = 16000
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.1  # applied to weight matrices only, not to norm gains
    beta2: float = 0.99  # AdamW's second-moment decay
    lr_schedule: str = "cosine"  # constant | cosine (decays from lr to min_lr over max_steps)
    min_lr: float = 3e-5
    grad_clip: float = 1.0  # max gradient norm; 0 = no clipping
    max_steps: int = 4000
    eval_every: int = 250
    early_stop_patience: int = 5  # evals without validation improvement
    # Patience only counts once validation accuracy has reached this level. Without it, a run
    # is stopped during the initial plateau, before the model has learned anything.
    early_stop_min_accuracy: float = 0.95
    n_probe_per_length: int = 200  # held-out instances per longer length, logged during training only
    # Pilots only: draw new training instances at every step instead of a fixed set of
    # n_train_instances. Answers "is this task size learnable at all?"; not a sample-efficiency run.
    fresh_data: bool = False
    seed: int = 0
    device: str = "auto"  # auto | cpu | mps | cuda


@dataclass
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    results_dir: str = "results"
