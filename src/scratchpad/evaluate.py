"""Step-by-step solving with the model's own states, and scoring against the oracle."""
from __future__ import annotations

from collections import Counter

import torch

from .formats import build_context, true_position_ids
from .task import Instance, State, oracle_trace
from .tokens import STATE_LEN, parse_state


@torch.no_grad()
def greedy_decode_batch(
    model, contexts: list[list[int]], n_tokens: int, position_ids: list[list[int]] | None = None
) -> list[list[int]]:
    """Generate n_tokens ids greedily after each context. All contexts must have the same length.

    Greedy = always take the most likely next token. Equal lengths mean no padding is needed;
    this holds whenever the instances share a node count and are at the same step.
    position_ids, if given, holds one position per context token plus n_tokens more (see
    formats.true_position_ids); None means the usual 0, 1, 2, ...
    """
    if len({len(c) for c in contexts}) != 1:
        raise ValueError("greedy_decode_batch needs contexts of equal length")
    device = next(model.parameters()).device
    ids = torch.tensor(contexts, dtype=torch.long, device=device)
    start = ids.shape[1]
    positions = None if position_ids is None else torch.tensor(position_ids, dtype=torch.long, device=device)
    for _ in range(n_tokens):
        kwargs = {} if positions is None else {"position_ids": positions[:, : ids.shape[1]]}
        logits = model(input_ids=ids, **kwargs).logits[:, -1, :]
        ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
    return ids[:, start:].tolist()


def greedy_decode(model, context: list[int], n_tokens: int) -> list[int]:
    """Generate n_tokens ids greedily after context."""
    return greedy_decode_batch(model, [context], n_tokens)[0]


def solve_batch(
    model,
    instances: list[Instance],
    k: int | None,
    max_steps: int | None = None,
    batch_size: int = 256,
    true_positions: bool = False,
) -> list[list[State | None]]:
    """Solve many instances in lockstep. See solve for the per-instance behaviour.

    max_steps=None caps the walk at the number of nodes: no correct walk is longer than that.
    """
    if max_steps is None:
        max_steps = max(len(inst.edges) for inst in instances)
    was_training = model.training
    model.eval()
    traces: list[list[State | None]] = [[] for _ in instances]
    active = list(range(len(instances)))
    for _ in range(max_steps):
        if not active:
            break
        still_active: list[int] = []
        for lo in range(0, len(active), batch_size):
            chunk = active[lo : lo + batch_size]
            # Invariant 4: the context is rebuilt from the model's own earlier states.
            contexts = [build_context(instances[i], traces[i], k)[0] for i in chunk]
            positions = None
            if true_positions:
                n_question = len(build_context(instances[chunk[0]], [], k)[0])
                positions = [true_position_ids(n_question, len(traces[i]), k) for i in chunk]
            outputs = greedy_decode_batch(model, contexts, STATE_LEN, positions)
            for i, out in zip(chunk, outputs):
                state = parse_state(out)
                traces[i].append(state)
                if state is not None and state[1] == "GO":
                    still_active.append(i)
        active = still_active
    model.train(was_training)
    return traces


def solve(
    model, inst: Instance, k: int | None, max_steps: int | None = None, true_positions: bool = False
) -> list[State | None]:
    """Generate states one at a time, rebuilding the context from the model's OWN states.

    Stops at YES/NO, at a format failure (None), or after max_steps (default: the node count).
    """
    return solve_batch(model, [inst], k, max_steps, true_positions=true_positions)[0]


def score(predicted: list[State | None], oracle: list[State]) -> dict:
    """final_correct, trace_correct, first_error_step (None if correct), format_failure."""
    trace_correct = predicted == oracle
    first_error_step: int | None = None
    if not trace_correct:
        first_error_step = next(
            (i for i, (p, o) in enumerate(zip(predicted, oracle)) if p != o),
            min(len(predicted), len(oracle)),  # one trace is a strict prefix of the other
        )
    last = predicted[-1] if predicted else None
    final_correct = last is not None and last[1] != "GO" and last[1] == oracle[-1][1]
    return {
        "final_correct": final_correct,
        "trace_correct": trace_correct,
        "first_error_step": first_error_step,
        "format_failure": any(p is None for p in predicted),
    }


def evaluate(model, eval_sets: dict[int, list[Instance]], k: int | None, true_positions: bool = False) -> dict:
    """Aggregate metrics per test length (plan.md §4)."""
    report: dict = {}
    for length, instances in eval_sets.items():
        predictions = solve_batch(model, instances, k, true_positions=true_positions)
        scores = [score(p, oracle_trace(inst)) for p, inst in zip(predictions, instances)]
        n = len(scores)
        errors = Counter(s["first_error_step"] for s in scores if s["first_error_step"] is not None)
        report[length] = {
            "n": n,
            "final_accuracy": sum(s["final_correct"] for s in scores) / n,
            "trace_accuracy": sum(s["trace_correct"] for s in scores) / n,
            "format_failure_rate": sum(s["format_failure"] for s in scores) / n,
            "first_error_hist": {str(step): count for step, count in sorted(errors.items())},
        }
    return report
