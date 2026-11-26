import random

from scratchpad.config import TaskConfig
from scratchpad.data import ExampleDataset, collate, keys_of, make_eval_sets, make_val_set
from scratchpad.task import instance_key, make_dataset, oracle_trace
from scratchpad.tokens import PAD_ID, STATE_LEN

SMALL = TaskConfig(n_eval_per_length=40, n_val=20)


def test_dataset_has_one_example_per_step():
    instances = make_dataset(10, 24, 6, random.Random(0))
    assert len(ExampleDataset(instances, 1)) == 10 * 6


def test_collate_puts_loss_only_on_target_tokens():
    # Invariant 5. Full history gives examples of different lengths, so padding is exercised.
    instances = make_dataset(4, 24, 6, random.Random(0))
    examples = ExampleDataset(instances, None).examples
    batch = collate(examples)
    for row, ex in enumerate(examples):
        n_context, n_total = len(ex.context), len(ex.context) + STATE_LEN
        labels = batch["labels"][row].tolist()
        assert labels[:n_context] == [-100] * n_context
        assert labels[n_context:n_total] == ex.target
        assert all(label == -100 for label in labels[n_total:])
        assert batch["input_ids"][row, :n_total].tolist() == ex.context + ex.target
        assert all(tok == PAD_ID for tok in batch["input_ids"][row, n_total:].tolist())
        assert batch["attention_mask"][row].tolist() == [1] * n_total + [0] * (len(labels) - n_total)


def test_eval_sets_have_right_lengths_and_balance():
    eval_sets = make_eval_sets(SMALL, exclude=set())
    assert set(eval_sets) == {3, 4, 5, 6}
    for length, instances in eval_sets.items():
        assert len(instances) == 40
        assert sum(inst.reachable for inst in instances) == 20
        assert all(len(oracle_trace(inst)) == length for inst in instances)


def test_eval_sets_are_fixed_across_calls():
    a = make_eval_sets(SMALL, exclude=set())
    b = make_eval_sets(SMALL, exclude=set())
    assert all(keys_of(a[n]) == keys_of(b[n]) for n in a)


def test_heldout_sets_skip_excluded_instances():
    # Invariant 6: exclude half of what would be generated and check none of it comes back.
    first = make_eval_sets(SMALL, exclude=set())
    excluded = {instance_key(inst) for inst in first[6][:20]}
    second = make_eval_sets(SMALL, exclude=excluded)
    assert keys_of(second[6]).isdisjoint(excluded)
    assert len(keys_of(second[6])) == 40


def test_val_set_is_disjoint_from_eval_sets():
    eval_sets = make_eval_sets(SMALL, exclude=set())
    eval_keys = {key for instances in eval_sets.values() for key in keys_of(instances)}
    val = make_val_set(SMALL, exclude=eval_keys)
    assert len(val) == 20
    assert keys_of(val).isdisjoint(eval_keys)


def test_fresh_stream_is_seeded_restricted_and_skips_excluded():
    import itertools

    from scratchpad.data import FreshExampleStream
    from scratchpad.tokens import STATE_LEN, encode_question

    def take(k, exclude=frozenset(), n=40):
        return list(itertools.islice(iter(FreshExampleStream(8, 3, k, 0, set(exclude))), n))

    inductive, full = take(1), take(None)
    # Same graphs and steps for every k (invariant 2); only the visible history differs.
    assert [(e.step, e.target, e.edge_spans) for e in inductive] == [(e.step, e.target, e.edge_spans) for e in full]
    assert {e.step for e in inductive} == {0, 1, 2}
    question_len = 8 * 4 + 4
    assert all(len(e.context) == question_len + STATE_LEN * min(e.step, 1) for e in inductive)
    assert all(len(e.context) == question_len + STATE_LEN * e.step for e in full)
    assert [e.context for e in take(1)] == [e.context for e in inductive]  # deterministic per seed
    assert [e.context for e in itertools.islice(iter(FreshExampleStream(8, 3, 1, 1, set())), 40)] != [
        e.context for e in inductive
    ]


def test_fresh_stream_never_yields_an_excluded_instance():
    import itertools
    import random

    from scratchpad.data import FreshExampleStream
    from scratchpad.formats import make_examples
    from scratchpad.task import instance_key, make_instance

    # Replay the stream's own generator to learn which instances it will draw, then exclude half.
    rng = random.Random("fresh-0")
    drawn, reachable = [], False
    for _ in range(20):
        reachable = not reachable
        drawn.append(make_instance(8, 3, reachable, rng))
        rng.randrange(3)
    excluded = {instance_key(inst) for inst in drawn[::2]}
    banned_contexts = {tuple(ex.context) for inst in drawn[::2] for ex in make_examples(inst, None)}
    got = list(itertools.islice(iter(FreshExampleStream(8, 3, None, 0, excluded)), 30))
    assert not any(tuple(e.context) in banned_contexts for e in got)


def test_collate_carries_position_ids_only_for_true_position_examples():
    import random

    from scratchpad.task import make_instance

    instances = [make_instance(8, 3, True, random.Random(i)) for i in range(4)]
    assert "position_ids" not in collate(ExampleDataset(instances, 1).examples)
    examples = ExampleDataset(instances, 1, true_positions=True).examples
    batch = collate(examples)
    assert batch["position_ids"].shape == batch["input_ids"].shape
    for row, ex in enumerate(examples):
        assert batch["position_ids"][row, : len(ex.position_ids)].tolist() == ex.position_ids
