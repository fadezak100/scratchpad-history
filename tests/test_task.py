import random

import pytest

from scratchpad.task import instance_key, make_dataset, make_instance, oracle_trace


def cycle_of(inst, start):
    """Nodes on the cycle through start, in walk order."""
    successor = dict(inst.edges)
    nodes = [start]
    while successor[nodes[-1]] != start:
        nodes.append(successor[nodes[-1]])
    return nodes


@pytest.mark.parametrize("n_steps", [6, 8, 12])
@pytest.mark.parametrize("reachable", [True, False])
def test_trace_has_exact_length_and_answer(n_steps, reachable):
    rng = random.Random(0)
    for _ in range(50):
        inst = make_instance(24, n_steps, reachable, rng)
        trace = oracle_trace(inst)
        assert len(trace) == n_steps
        assert all(status == "GO" for _, status in trace[:-1])
        assert trace[-1][1] == ("YES" if reachable else "NO")


def test_graph_is_disjoint_cycles():
    inst = make_instance(24, 6, True, random.Random(1))
    sources = [u for u, _ in inst.edges]
    targets = [v for _, v in inst.edges]
    assert sorted(sources) == list(range(24))  # one outgoing edge per node
    assert sorted(targets) == list(range(24))  # one incoming edge per node
    assert all(u != v for u, v in inst.edges)  # no self-loops


@pytest.mark.parametrize("n_steps", [6, 8, 10, 12])
def test_unreachable_is_two_cycles_of_L_and_rest(n_steps):
    rng = random.Random(4)
    for _ in range(20):
        inst = make_instance(24, n_steps, False, rng)
        source_cycle = cycle_of(inst, inst.source)
        target_cycle = cycle_of(inst, inst.target)
        assert len(source_cycle) == n_steps
        assert len(target_cycle) == 24 - n_steps
        assert set(source_cycle).isdisjoint(target_cycle)


@pytest.mark.parametrize("n_steps", [6, 8, 10, 12])
def test_reachable_is_one_full_cycle_with_target_L_ahead(n_steps):
    rng = random.Random(5)
    for _ in range(20):
        inst = make_instance(24, n_steps, True, rng)
        cycle = cycle_of(inst, inst.source)
        assert len(cycle) == 24
        assert cycle[n_steps] == inst.target


@pytest.mark.parametrize("n_steps", [0, 1, 23, 24])
def test_invalid_step_count_raises(n_steps):
    with pytest.raises(ValueError):
        make_instance(24, n_steps, True, random.Random(0))


def test_edges_are_shuffled():
    # Invariant 7: the source's edge must not sit at a fixed place in the edge list.
    rng = random.Random(6)
    positions = set()
    for _ in range(50):
        inst = make_instance(24, 6, True, rng)
        positions.add([u for u, _ in inst.edges].index(inst.source))
    assert len(positions) > 5


def test_dataset_is_balanced():
    data = make_dataset(200, 24, 6, random.Random(2))
    assert sum(inst.reachable for inst in data) == 100


def test_same_seed_same_data():
    a = make_dataset(20, 24, 6, random.Random(3))
    b = make_dataset(20, 24, 6, random.Random(3))
    assert [instance_key(x) for x in a] == [instance_key(x) for x in b]
