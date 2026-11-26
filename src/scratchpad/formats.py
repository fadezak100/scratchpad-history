"""Turn an instance into training examples for a given history length k. Pure Python.

Every condition produces the same kind of example: context -> next state, where
context = question + the last k oracle states (k=None: all of them). Because each
context starts with the question, positions reset for free (inductive re-indexing).

With true_positions=True the visible tokens keep the position numbers they would have in the
full sequence (question, then every state in order), so history is hidden but positions are
NOT reset. This is the control that separates the two effects of the inductive scratchpad.
"""
from __future__ import annotations

from dataclasses import dataclass

from .task import Instance, State, oracle_trace
from .tokens import EDGE_LEN, STATE_LEN, encode_question, encode_state


@dataclass(frozen=True)
class Example:
    context: list[int]
    target: list[int]  # encoded next state (STATE_LEN tokens)
    step: int  # index of the target state in the trace (0-based)
    current_node: int  # node the walk is at before this step
    next_node: int  # node the target state moves to
    edge_spans: dict[int, tuple[int, int]]  # source node -> [start, end) of its edge in context
    # Position number of every token of context + target; None = 0, 1, 2, ... (positions reset).
    position_ids: list[int] | None = None


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


def true_position_ids(n_question: int, n_previous: int, k: int | None) -> list[int]:
    """Positions of context + target when nothing is renumbered.

    The question keeps 0..n_question-1 and state i (0-based) keeps the STATE_LEN positions it has
    in the full sequence. Covers the visible previous states and the state about to be written.
    For k=None this is simply 0, 1, 2, ...
    """
    positions = list(range(n_question))
    for i in [*history_window(n_previous, k), n_previous]:
        start = n_question + i * STATE_LEN
        positions.extend(range(start, start + STATE_LEN))
    return positions


def make_examples(inst: Instance, k: int | None, true_positions: bool = False) -> list[Example]:
    """One example per oracle step."""
    trace = oracle_trace(inst)
    n_question = len(encode_question(inst))
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
                position_ids=true_position_ids(n_question, step, k) if true_positions else None,
            )
        )
        current = state[0]
    return examples
