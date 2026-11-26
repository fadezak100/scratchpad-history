import random

import pytest

from scratchpad.formats import make_examples
from scratchpad.task import make_instance, oracle_trace
from scratchpad.tokens import STATE_LEN, encode_edge, encode_question, encode_state


@pytest.fixture
def inst():
    return make_instance(24, 6, True, random.Random(0))


@pytest.mark.parametrize("k", [1, 2, 4, None])
def test_context_holds_last_k_states(inst, k):
    q_len = len(encode_question(inst))
    for ex in make_examples(inst, k):
        n_visible = ex.step if k is None else min(k, ex.step)
        assert len(ex.context) == q_len + n_visible * STATE_LEN


def test_one_example_per_step_with_oracle_targets(inst):
    trace = oracle_trace(inst)
    examples = make_examples(inst, None)
    assert [ex.target for ex in examples] == [encode_state(s) for s in trace]


def test_inductive_context_is_question_plus_previous_state(inst):
    trace = oracle_trace(inst)
    q = encode_question(inst)
    for ex in make_examples(inst, 1)[1:]:
        assert ex.context == q + encode_state(trace[ex.step - 1])


def test_edge_spans_point_at_current_edge(inst):
    for ex in make_examples(inst, None):
        start, end = ex.edge_spans[ex.current_node]
        assert ex.next_node == dict(inst.edges)[ex.current_node]
        assert ex.context[start:end] == encode_edge((ex.current_node, ex.next_node))


def test_true_positions_keep_full_sequence_numbering():
    import random

    from scratchpad.formats import true_position_ids
    from scratchpad.task import make_instance
    from scratchpad.tokens import STATE_LEN, encode_question

    inst = make_instance(8, 3, True, random.Random(0))
    q = len(encode_question(inst))
    # Full history: nothing is hidden, so true positions are just 0, 1, 2, ...
    for ex in make_examples(inst, None, true_positions=True):
        assert ex.position_ids == list(range(len(ex.context) + len(ex.target)))
    # k=1: the one visible state and the target sit where they would in the full sequence.
    for ex in make_examples(inst, 1, true_positions=True):
        assert len(ex.position_ids) == len(ex.context) + len(ex.target)
        assert ex.position_ids[:q] == list(range(q))
        assert ex.position_ids[-STATE_LEN:] == list(range(q + ex.step * STATE_LEN, q + (ex.step + 1) * STATE_LEN))
    assert true_position_ids(10, 3, 2) == list(range(10)) + [13, 14, 15, 16, 17, 18, 19, 20, 21]
    # Default: positions reset, no position ids carried.
    assert all(ex.position_ids is None for ex in make_examples(inst, 1))
