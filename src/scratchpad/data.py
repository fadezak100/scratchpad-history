"""Torch datasets, padding collate, and fixed held-out sets."""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset, IterableDataset

from .config import TaskConfig
from .formats import Example, make_examples
from .task import Instance, instance_key, make_instance
from .tokens import PAD_ID

IGNORE_INDEX = -100  # label value that the cross-entropy loss skips


class ExampleDataset(Dataset):
    """Flattened examples from a list of instances for one history k."""

    def __init__(self, instances: list[Instance], k: int | None, true_positions: bool = False):
        self.examples: list[Example] = [
            ex for inst in instances for ex in make_examples(inst, k, true_positions)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


class FreshExampleStream(IterableDataset):
    """Endless stream of examples, each from a newly generated instance whose key is not in exclude.

    One example per instance, at a uniformly random step, so a batch has the same mix of steps as a
    shuffled ExampleDataset. The graphs and steps depend on the seed only, never on k (invariant 2).
    """

    def __init__(
        self, n_nodes: int, n_steps: int, k: int | None, seed: int, exclude: set, true_positions: bool = False
    ):
        self.n_nodes, self.n_steps, self.k, self.seed, self.exclude = n_nodes, n_steps, k, seed, exclude
        self.true_positions = true_positions

    def __iter__(self):
        rng = random.Random(f"fresh-{self.seed}")
        reachable = False
        while True:
            reachable = not reachable
            inst = make_instance(self.n_nodes, self.n_steps, reachable, rng)
            step = rng.randrange(self.n_steps)  # drawn before the exclude check so the stream is the same for every k
            if instance_key(inst) in self.exclude:
                continue
            yield make_examples(inst, self.k, self.true_positions)[step]


def collate(batch: list[Example]) -> dict[str, torch.Tensor]:
    """Right-pad to the longest example.

    input_ids = context + target; labels = -100 on context and padding, target ids otherwise.
    Returns input_ids, attention_mask, labels, and position_ids when the examples carry them
    (padding gets position 0; it is masked out anyway).
    """
    max_len = max(len(ex.context) + len(ex.target) for ex in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, ex in enumerate(batch):
        n_context = len(ex.context)
        n_total = n_context + len(ex.target)
        input_ids[row, :n_total] = torch.tensor(ex.context + ex.target, dtype=torch.long)
        attention_mask[row, :n_total] = 1
        labels[row, n_context:n_total] = torch.tensor(ex.target, dtype=torch.long)
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if batch[0].position_ids is not None:
        position_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, ex in enumerate(batch):
            position_ids[row, : len(ex.position_ids)] = torch.tensor(ex.position_ids, dtype=torch.long)
        out["position_ids"] = position_ids
    return out


def make_heldout(
    n_instances: int, n_nodes: int, n_steps: int, rng: random.Random, exclude: set
) -> list[Instance]:
    """Balanced YES/NO instances whose keys are not in exclude and not repeated within the set."""
    seen: set = set()
    instances: list[Instance] = []
    for i in range(n_instances):
        reachable = i % 2 == 0
        while True:
            inst = make_instance(n_nodes, n_steps, reachable, rng)
            key = instance_key(inst)
            if key not in exclude and key not in seen:
                break
        seen.add(key)
        instances.append(inst)
    return instances


def make_eval_sets(cfg: TaskConfig, exclude: set) -> dict[int, list[Instance]]:
    """Held-out instances per test length, generated from cfg.eval_seed, skipping keys in exclude.

    Each length gets its own RNG, so a set doesn't change when other lengths are added or removed.
    """
    return {
        length: make_heldout(
            cfg.n_eval_per_length,
            cfg.n_nodes,
            length,
            random.Random(f"eval-{cfg.eval_seed}-{length}"),
            exclude,
        )
        for length in cfg.test_steps
    }


def make_val_set(cfg: TaskConfig, exclude: set) -> list[Instance]:
    """Validation instances at the training length, used only for early stopping.

    Pass the training keys plus the eval-set keys as exclude, so validation overlaps neither.
    """
    rng = random.Random(f"val-{cfg.val_seed}")
    return make_heldout(cfg.n_val, cfg.n_nodes, cfg.train_steps, rng, exclude)


def keys_of(instances: list[Instance]) -> set:
    return {instance_key(inst) for inst in instances}
