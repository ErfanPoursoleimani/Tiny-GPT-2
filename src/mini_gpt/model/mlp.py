"""Position-wise feed-forward network: Linear(C,4C) -> GELU -> Linear(4C,C)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.config import ModelConfig


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.embedding_dim, 4 * cfg.embedding_dim, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.embedding_dim, cfg.embedding_dim, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        # "gelu" = exact erf-based GELU; "gelu_new" = tanh approximation used by the original GPT-2.
        self.approximate = "tanh" if cfg.activation == "gelu_new" else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x), approximate=self.approximate)
        return self.drop(self.c_proj(x))
