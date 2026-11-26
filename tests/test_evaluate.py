import random

import torch

from scratchpad import evaluate as ev
from scratchpad.config import ModelConfig
from scratchpad.formats import build_context
from scratchpad.model import build_model
from scratchpad.task import make_instance, oracle_trace
from scratchpad.tokens import STATE_LEN, VOCAB, encode_state

ORACLE = [(3, "GO"), (5, "GO"), (9, "YES")]


class DummyModel(torch.nn.Module):
    """Stands in for a model when the decoder is replaced by a script."""


def test_score_correct_trace():
    result = ev.score(list(ORACLE), ORACLE)
    assert result == {
        "final_correct": True,
        "trace_correct": True,
        "first_error_step": None,
        "format_failure": False,
    }


def test_score_wrong_node_but_right_answer():
    result = ev.score([(3, "GO"), (4, "GO"), (9, "YES")], ORACLE)
    assert result["final_correct"] and not result["trace_correct"]
    assert result["first_error_step"] == 1


def test_score_stops_too_early():
    result = ev.score([(3, "GO"), (5, "NO")], ORACLE)
    assert not result["final_correct"]
    assert result["first_error_step"] == 1


def test_score_format_failure():
    result = ev.score([(3, "GO"), None], ORACLE)
    assert result["format_failure"] and not result["final_correct"]
    assert result["first_error_step"] == 1


def test_score_never_terminates():
    # The walk kept saying GO until the step cap: a wrong answer, first wrong at the last oracle step.
    result = ev.score([(3, "GO"), (5, "GO"), (9, "GO"), (1, "GO")], ORACLE)
    assert not result["final_correct"]
    assert result["first_error_step"] == 2


def test_solve_feeds_back_the_models_own_states(monkeypatch):
    # Invariant 4: script a wrong first state and check it, not the oracle's, appears in the next context.
    inst = make_instance(24, 6, True, random.Random(0))
    oracle = oracle_trace(inst)
    wrong_node = next(n for n in range(24) if n not in (oracle[0][0], inst.source, inst.target))
    script = [(wrong_node, "GO"), (inst.target, "YES")]
    seen_contexts = []

    def fake_decode(model, contexts, n_tokens, position_ids=None):
        seen_contexts.append(contexts[0])
        return [encode_state(script[len(seen_contexts) - 1])]

    monkeypatch.setattr(ev, "greedy_decode_batch", fake_decode)
    predicted = ev.solve(DummyModel(), inst, k=1)
    assert predicted == script
    assert seen_contexts[1] == build_context(inst, [script[0]], 1)[0]
    assert seen_contexts[1] != build_context(inst, [oracle[0]], 1)[0]


def test_solve_stops_on_format_failure_and_on_step_cap(monkeypatch):
    inst = make_instance(24, 6, True, random.Random(0))
    monkeypatch.setattr(ev, "greedy_decode_batch", lambda m, c, n, p=None: [[0, 0, 0]] * len(c))
    assert ev.solve(DummyModel(), inst, k=None) == [None]
    monkeypatch.setattr(ev, "greedy_decode_batch", lambda m, c, n, p=None: [encode_state((2, "GO"))] * len(c))
    assert ev.solve(DummyModel(), inst, k=None, max_steps=7) == [(2, "GO")] * 7


def test_untrained_model_runs_end_to_end():
    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_heads=2), len(VOCAB))
    instances = [make_instance(24, 6, reachable, random.Random(1)) for reachable in (True, False)]
    out = ev.greedy_decode(model, build_context(instances[0], [], None)[0], STATE_LEN)
    assert len(out) == STATE_LEN and all(0 <= tok < len(VOCAB) for tok in out)
    report = ev.evaluate(model, {6: instances}, k=1)
    assert report[6]["n"] == 2
    assert 0.0 <= report[6]["final_accuracy"] <= 1.0


def test_solve_with_true_positions_passes_full_sequence_positions(monkeypatch):
    # Position-reset control: at step 2 with k=1 the visible state and the target keep the positions
    # they have in the full sequence, so there is a gap after the question.
    from scratchpad.tokens import STATE_LEN, encode_question

    inst = make_instance(24, 6, True, random.Random(0))
    oracle = oracle_trace(inst)
    seen_positions = []

    def fake_decode(model, contexts, n_tokens, position_ids=None):
        seen_positions.append(position_ids[0])
        assert len(position_ids[0]) == len(contexts[0]) + n_tokens
        return [encode_state(oracle[len(seen_positions) - 1])]

    monkeypatch.setattr(ev, "greedy_decode_batch", fake_decode)
    assert ev.solve(DummyModel(), inst, k=1, true_positions=True) == oracle
    q = len(encode_question(inst))
    assert seen_positions[0] == list(range(q + STATE_LEN))  # first step: nothing hidden yet
    assert seen_positions[2] == list(range(q)) + list(range(q + STATE_LEN, q + 3 * STATE_LEN))
