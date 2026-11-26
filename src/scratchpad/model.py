"""Tiny randomly initialized Llama (RoPE positions) for next-state prediction."""
from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM

from .config import ModelConfig
from .tokens import EOS_ID, PAD_ID


def build_model(cfg: ModelConfig, vocab_size: int) -> LlamaForCausalLM:
    """LlamaConfig from cfg. The default eager attention is what makes output_attentions work."""
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_heads,
        max_position_embeddings=cfg.max_positions,
        pad_token_id=PAD_ID,
        bos_token_id=None,
        eos_token_id=EOS_ID,
        attn_implementation=cfg.attn_implementation,
    )
    return LlamaForCausalLM(config)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
