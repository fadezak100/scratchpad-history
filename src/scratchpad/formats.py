"""Turn an instance into training examples for a given history length k. Pure Python.

Every condition produces the same kind of example: context -> next state, where
context = question + the last k oracle states (k=None: all of them). Because each
context starts with the question, positions reset for free (inductive re-indexing).
"""
from __future__ import annotations

from dataclasses import dataclass

from .task import Instance, State, oracle_trace
from .tokens import EDGE_LEN, encode_question, encode_state


@dataclass(frozen=True)
class Example:
    context: list[int]
    target: list[int]  # encoded next state (STATE_LEN tokens)
    step: int  # index of the target state in the trace (0-based)
    current_node: int  # node the walk is at before this step
    next_node: int  # node the target state moves to
    edge_spans: dict[int, tuple[int, int]]  # source node -> [start, end) of its edge in context


def history_window(n_previous: int, k: int | None) -> range:
    """Indices of the previous states visible under history k."""
    if k is None:
        return range(n_previous)
    if k < 1:
        raise ValueError("k must be >= 1, or None for full history")
    return range(max(0, n_previous - k), n_previous)


def edge_spans(inst: Instance) -> dict[int, tuple[int, int]]:
    """Source node -> [start, end) of its edge's tokens. Edges come first, so spans don't depend on k."""
    return {u: (i * EDGE_LEN, (i + 1) * EDGE_LEN) for i, (u, _) in enumerate(inst.edges)}


def build_context(
    inst: Instance, previous_states: list[State], k: int | None
) -> tuple[list[int], dict[int, tuple[int, int]]]:
    """Question + visible previous states, plus edge spans. Shared by training and evaluate.solve."""
    context = encode_question(inst)
    for i in history_window(len(previous_states), k):
        context.extend(encode_state(previous_states[i]))
    return context, edge_spans(inst)


def make_examples(inst: Instance, k: int | None) -> list[Example]:
    """One example per oracle step."""
    trace = oracle_trace(inst)
    examples: list[Example] = []
    current = inst.source
    for step, state in enumerate(trace):
        context, spans = build_context(inst, trace[:step], k)
        examples.append(
            Example(
                context=context,
                target=encode_state(state),
                step=step,
                current_node=current,
                next_node=state[0],
                edge_spans=spans,
            )
        )
        current = state[0]
    return examples
