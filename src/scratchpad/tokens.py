"""Fixed vocabulary and encode/decode. Pure Python, no torch.

Layout of one instance (edges shuffled):
    3 > 7 ; 7 > 1 ; ... Q 3 9 |  S 7 GO  S 1 GO  ...  S 9 YES
"""
from __future__ import annotations

from .task import Edge, Instance, State

N_NODES = 24
SPECIAL = ["<pad>", "<eos>", ">", ";", "Q", "|", "S", "GO", "YES", "NO"]
VOCAB: list[str] = SPECIAL + [str(i) for i in range(N_NODES)]
TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID["<pad>"]
EOS_ID = TOKEN_TO_ID["<eos>"]
STATE_LEN = 3  # "S", node, status
EDGE_LEN = 4  # node, ">", node, ";"
STATUSES = ("GO", "YES", "NO")
FIRST_NODE_ID = len(SPECIAL)  # node n has id FIRST_NODE_ID + n


def node_id(node: int) -> int:
    return TOKEN_TO_ID[str(node)]


def encode_edge(edge: Edge) -> list[int]:
    """Edge (u, v) -> ids for "u > v ;"."""
    u, v = edge
    return [node_id(u), TOKEN_TO_ID[">"], node_id(v), TOKEN_TO_ID[";"]]


def encode_question(inst: Instance) -> list[int]:
    """All edges in instance order, then "Q s t |"."""
    ids: list[int] = []
    for edge in inst.edges:
        ids.extend(encode_edge(edge))
    ids.extend([TOKEN_TO_ID["Q"], node_id(inst.source), node_id(inst.target), TOKEN_TO_ID["|"]])
    return ids


def encode_state(state: State) -> list[int]:
    """(node, status) -> ids for "S node status"."""
    node, status = state
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    return [TOKEN_TO_ID["S"], node_id(node), TOKEN_TO_ID[status]]


def parse_state(ids: list[int]) -> State | None:
    """Inverse of encode_state; None if ids aren't a well-formed state (a format failure)."""
    if len(ids) != STATE_LEN or ids[0] != TOKEN_TO_ID["S"]:
        return None
    node_tok, status_tok = ids[1], ids[2]
    if not FIRST_NODE_ID <= node_tok < len(VOCAB):
        return None
    if not 0 <= status_tok < len(VOCAB) or VOCAB[status_tok] not in STATUSES:
        return None
    return (node_tok - FIRST_NODE_ID, VOCAB[status_tok])


def decode(ids: list[int]) -> str:
    """Ids -> space-separated tokens, for debugging."""
    return " ".join(VOCAB[i] for i in ids)
