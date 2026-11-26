from __future__ import annotations

import random

import pytest
import torch

from scratchpad import attention as att
from scratchpad.config import ModelConfig
from scratchpad.formats import make_examples
from scratchpad.model import build_model
from scratchpad.task import make_instance
from scratchpad.tokens import EDGE_LEN, VOCAB


def tiny_model():
    torch.manual_seed(0)
    return build_model(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=2, num_heads=2), len(VOCAB))


def examples(k, n_steps=3):
    inst = make_instance(8, n_steps, True, random.Random(0))
    return make_examples(inst, k)


def test_uniform_attention_gives_chance():
    ex = examples(None)[-1]
    attn = torch.full((len(ex.context) + 1,), 1.0)
    share = att.share_from_attention(attn, ex.edge_spans, ex.current_node)
    assert share.item() == pytest.approx(1 / 8)


def test_all_attention_on_correct_edge_gives_one():
    ex = examples(None)[-1]
    attn = torch.zeros(len(ex.context) + 1)
    lo, hi = ex.edge_spans[ex.current_node]
    attn[lo:hi] = 0.1
    attn[-1] = 0.6  # weight outside the edges (history, the question) must not change the share
    assert att.share_from_attention(attn, ex.edge_spans, ex.current_node).item() == pytest.approx(1.0)


def test_history_tokens_do_not_change_the_share():
    # Same weights on the edges; the full-history context just has extra positions after them.
    short, full = examples(1)[-1], examples(None)[-1]
    assert len(full.context) > len(short.context)
    n_edge_tokens = 8 * EDGE_LEN
    edge_weights = torch.rand(n_edge_tokens, generator=torch.Generator().manual_seed(0))
    shares = []
    for ex in (short, full):
        attn = torch.full((len(ex.context) + 1,), 0.5)
        attn[:n_edge_tokens] = edge_weights
        shares.append(att.share_from_attention(attn, ex.edge_spans, ex.current_node).item())
    assert shares[0] == pytest.approx(shares[1])


def test_shares_over_all_edges_sum_to_one():
    model = tiny_model()
    ex = examples(1)[0]
    total = sum(
        att.edge_attention_shares(model, [ex.__class__(**{**ex.__dict__, "current_node": node})])[0]
        for node in ex.edge_spans
    )
    assert torch.allclose(total, torch.ones(2, 2), atol=1e-5)


def test_single_matches_batched_and_has_layer_head_shape():
    model = tiny_model()
    exs = examples(None, n_steps=4)  # contexts of different lengths
    batched = att.edge_attention_shares(model, exs)
    assert batched.shape == (len(exs), 2, 2)
    for i, ex in enumerate(exs):
        assert torch.allclose(att.edge_attention_share(model, ex), batched[i], atol=1e-5)


def test_report_structure():
    model = tiny_model()
    exs = examples(None)
    report = att.attention_report(model, exs)
    assert report["chance"] == pytest.approx(1 / 8)
    assert set(report["by_step"]) == {"0", "1", "2"}
    best = report["best_head"]
    assert report["mean"][best["layer"]][best["head"]] == pytest.approx(best["share"])
    assert best["share"] == pytest.approx(max(max(row) for row in report["mean"]))
