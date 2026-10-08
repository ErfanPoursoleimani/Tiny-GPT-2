"""Token + positional embeddings, and Rotary Positional Embeddings (optional extension).

Shapes:  idx (B, T) -> x (B, T, C)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mini_gpt.config import ModelConfig


class GPTEmbeddings(nn.Module):
    """x = Dropout(TokenEmb(idx) + PosEmb(positions)).  With RoPE there is no additive position term."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embedding_dim) if cfg.position_encoding == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        B, T = idx.shape
        if pos_offset + T > self.cfg.context_length:
            raise ValueError(
                f"Sequence positions {pos_offset}..{pos_offset + T} exceed context_length={self.cfg.context_length}"
            )
        x = self.tok_emb(idx)  # (B, T, C)
        if self.pos_emb is not None:
            pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)  # (T,)
            x = x + self.pos_emb(pos)  # broadcast (T, C) over batch
        return self.drop(x)


class RotaryEmbedding(nn.Module):
    """RoPE (Su et al., 2021): rotate each (x_i, x_{i+D/2}) pair of q/k by an angle proportional to position.

    The dot product q_m . k_n then depends only on the relative offset m - n.
    Applied to q and k of shape (B, H, T, D); cos/sin tables are (max_len, D/2).
    """

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_len, D/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Rotate in >= float32 (half-precision inputs are upcast; float32/float64 are kept as-is).
        work = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        x1, x2 = x.to(work).chunk(2, dim=-1)
        cos, sin = cos.to(work), sin.to(work)
        out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.to(x.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(2)
        cos = self.cos[offset : offset + T].to(q.device)  # (T, D/2) broadcasts over (B, H)
        sin = self.sin[offset : offset + T].to(q.device)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)
