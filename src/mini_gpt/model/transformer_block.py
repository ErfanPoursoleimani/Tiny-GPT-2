"""Pre-LayerNorm Transformer block (GPT-2 style)."""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_gpt.config import ModelConfig
from mini_gpt.model.attention import CausalSelfAttention, KVCache
from mini_gpt.model.mlp import MLP


class TransformerBlock(nn.Module):
    """x = x + Attn(LN(x));  x = x + MLP(LN(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.embedding_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.embedding_dim)
        self.mlp = MLP(cfg)

    def forward(
        self, x: torch.Tensor, past_kv: KVCache | None = None, use_cache: bool = False, pos_offset: int = 0
    ) -> tuple[torch.Tensor, KVCache | None]:
        a, present = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache, pos_offset=pos_offset)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present
