"""Cycle-reachability instances and the oracle trace. Pure Python, no torch.

Graph construction follows arXiv 2406.06467 §3.2.1 (see plan.md §2). For an instance that
needs L steps on n nodes:

    NO:  two disjoint cycles of length L and n - L; s is in the L-cycle, t in the other.
    YES: one cycle through all n nodes; t is exactly L steps ahead of s.

Every node has exactly one outgoing edge, so the walk from s is deterministic. It stops at
t (YES) or when it returns to s (NO).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

Edge = tuple[int, int]
State = tuple[int, str]  # (current node, status) with status in {"GO", "YES", "NO"}


@dataclass(frozen=True)
class Instance:
    edges: tuple[Edge, ...]  # shuffled
    source: int
    target: int
    n_steps: int
    reachable: bool


def cycle_edges(nodes: list[int]) -> list[Edge]:
    """Edges of the directed cycle nodes[0] -> nodes[1] -> ... -> nodes[-1] -> nodes[0]."""
    if len(nodes) < 2:
        raise ValueError("a cycle needs at least 2 nodes")
    return [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]


def make_instance(n_nodes: int, n_steps: int, reachable: bool, rng: random.Random) -> Instance:
    """Build an instance whose oracle trace has exactly n_steps states.

    reachable=True: one n_nodes-cycle, t is n_steps ahead of s.
    reachable=False: cycles of length n_steps and n_nodes - n_steps; s in the first, t in the second.
    """
    if not 2 <= n_steps <= n_nodes - 2:
        raise ValueError(f"n_steps must be in [2, {n_nodes - 2}], got {n_steps}")
    order = list(range(n_nodes))
    rng.shuffle(order)
    source = order[0]
    if reachable:
        edges = cycle_edges(order)
        target = order[n_steps]
    else:
        edges = cycle_edges(order[:n_steps]) + cycle_edges(order[n_steps:])
        target = rng.choice(order[n_steps:])
    rng.shuffle(edges)
    return Instance(tuple(edges), source, target, n_steps, reachable)


def oracle_trace(inst: Instance) -> list[State]:
    """Follow edges from inst.source; return one state per step, last status YES or NO."""
    successor = dict(inst.edges)
    trace: list[State] = []
    node = inst.source
    for _ in range(len(inst.edges)):
        node = successor[node]
        if node == inst.target:
            trace.append((node, "YES"))
            return trace
        if node == inst.source:
            trace.append((node, "NO"))
            return trace
        trace.append((node, "GO"))
    raise ValueError("walk did not terminate; the graph is not a set of disjoint cycles")


def make_dataset(
    n_instances: int, n_nodes: int, n_steps: int, rng: random.Random
) -> list[Instance]:
    """Balanced YES/NO list of instances (in random order), all with exactly n_steps steps."""
    labels = [i % 2 == 0 for i in range(n_instances)]
    rng.shuffle(labels)
    return [make_instance(n_nodes, n_steps, reachable, rng) for reachable in labels]


def instance_key(inst: Instance) -> tuple:
    """Hashable identity (sorted edges + query) used to keep held-out sets disjoint from training."""
    return (tuple(sorted(inst.edges)), inst.source, inst.target)
