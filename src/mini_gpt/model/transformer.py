"""Stack of Transformer blocks followed by the final LayerNorm."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from mini_gpt.config import ModelConfig
from mini_gpt.model.attention import KVCache
from mini_gpt.model.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.num_layers))
        self.ln_f = nn.LayerNorm(cfg.embedding_dim)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        past_kvs: list[KVCache | None] | None = None,
        use_cache: bool = False,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        presents: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                # Recompute this block's activations during backward instead of storing them.
                x = checkpoint(lambda t, b=block: b(t)[0], x, use_reentrant=False)
            else:
                x, present = block(x, past_kv=past, use_cache=use_cache, pos_offset=pos_offset)
                if use_cache:
                    presents.append(present)
        return self.ln_f(x), (presents if use_cache else None)
