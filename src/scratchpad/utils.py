"""Seeding, device selection, run IDs, JSON I/O."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_id(k: int | None, n_train: int, seed: int, true_positions: bool = False) -> str:
    """e.g. k1_n16000_s0; "tp" after k marks a true-positions (no position reset) run."""
    return f"k{'full' if k is None else k}{'tp' if true_positions else ''}_n{n_train}_s{seed}"


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(obj) if is_dataclass(obj) else obj
    path.write_text(json.dumps(data, indent=2))


def first_reaching(history: list[dict], threshold: float, key: str = "val_trace_accuracy") -> dict | None:
    """The first logged evaluation whose `key` is at least `threshold`, or None if none is.

    `history` is the list saved in metrics.json. Evaluations happen every eval_every steps, so the
    step, examples and tokens returned are rounded up to the next evaluation.
    """
    for entry in history:
        if entry[key] >= threshold:
            return entry
    return None
