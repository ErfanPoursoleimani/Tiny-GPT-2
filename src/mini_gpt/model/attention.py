"""Multi-head causal self-attention (manual / SDPA / auto) with optional KV cache.

Tensor shapes (B=batch, T=query length, S=key length incl. cache, C=embedding, H=heads, D=C/H):

    x                : (B, T, C)
    qkv = c_attn(x)  : (B, T, 3C)    -> split -> q, k, v : (B, T, C)
    q, k, v (heads)  : (B, H, T, D)  (k, v: (B, H, S, D) once concatenated with the cache)
    scores           : (B, H, T, S)  = q @ k^T / sqrt(D)
    causal mask      : (T, S)        True where key j <= absolute query position
    y                : (B, H, T, D) -> (B, T, C) -> c_proj -> (B, T, C)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.config import ModelConfig
from mini_gpt.model.embeddings import RotaryEmbedding

KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.num_heads
        self.n_embd = cfg.embedding_dim
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout
        self.c_attn = nn.Linear(cfg.embedding_dim, 3 * cfg.embedding_dim, bias=cfg.bias)  # fused W_Q, W_K, W_V
        self.c_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=cfg.bias)  # W_O
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.context_length) if cfg.position_encoding == "rope" else None
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        self.impl = cfg.attention_impl
        self.store_attention = False  # tests/inspection: keep last attention probabilities (manual only)
        self.last_attention: torch.Tensor | None = None

    def resolved_impl(self) -> str:
        if self.impl == "auto":
            return "sdpa" if hasattr(F, "scaled_dot_product_attention") else "manual"
        if self.impl == "sdpa" and not hasattr(F, "scaled_dot_product_attention"):
            raise RuntimeError("attention.implementation=sdpa requires PyTorch >= 2.0")
        return self.impl

    def forward(
        self, x: torch.Tensor, past_kv: KVCache | None = None, use_cache: bool = False, pos_offset: int = 0
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape
        assert C == self.n_embd, f"expected last dim {self.n_embd}, got {C}"
        H, D = self.n_head, self.head_dim

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # each (B, T, C)
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        assert q.shape == (B, H, T, D)

        if self.rope is not None:
            q, k = self.rope(q, k, offset=pos_offset)  # rotate *before* caching so cached keys are final

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)  # (B, H, S, D)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None
        S = k.size(2)
        if S > self.causal_mask.size(0):
            raise ValueError(f"Key length {S} exceeds context_length {self.causal_mask.size(0)}")

        if self.resolved_impl() == "sdpa":
            p = self.dropout_p if self.training else 0.0
            if T == S:  # no cache (or cache empty): standard causal attention
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
            elif T == 1:  # single new token attends to the whole cache: no mask needed
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            else:  # chunk appended to a cache: offset causal mask
                mask = self.causal_mask[S - T : S, :S]
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, S)
            assert att.shape == (B, H, T, S)
            mask = self.causal_mask[S - T : S, :S]  # (T, S): row i may see columns <= S-T+i
            att = att.masked_fill(~mask, float("-inf"))
            att = att.softmax(dim=-1)
            if self.store_attention:
                self.last_attention = att.detach()
            att = self.attn_drop(att)
            y = att @ v  # (B, H, T, D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # concat heads
        return self.resid_drop(self.c_proj(y)), present
