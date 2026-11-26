import random

from scratchpad.task import make_instance
from scratchpad.tokens import STATE_LEN, VOCAB, encode_question, encode_state, parse_state


def test_state_round_trip():
    for state in [(0, "GO"), (23, "YES"), (7, "NO")]:
        ids = encode_state(state)
        assert len(ids) == STATE_LEN
        assert parse_state(ids) == state


def test_parse_rejects_malformed():
    assert parse_state(encode_state((3, "GO"))[:2]) is None
    assert parse_state([VOCAB.index("GO")] * STATE_LEN) is None


def test_question_contains_all_edges_and_query():
    inst = make_instance(24, 6, True, random.Random(0))
    ids = encode_question(inst)
    assert ids.count(VOCAB.index(">")) == 24
    assert ids[-4:] == [VOCAB.index("Q"), VOCAB.index(str(inst.source)),
                        VOCAB.index(str(inst.target)), VOCAB.index("|")]
