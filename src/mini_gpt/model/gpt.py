"""GPT: embeddings -> Transformer -> LM head, with GPT-style init, loss and weight tying."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_gpt.config import ModelConfig
from mini_gpt.model.attention import KVCache
from mini_gpt.model.embeddings import GPTEmbeddings
from mini_gpt.model.transformer import Transformer


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embeddings = GPTEmbeddings(cfg)
        self.transformer = Transformer(cfg)
        self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embeddings.tok_emb.weight  # W_lm_head = W_token_embedding
        self.apply(self._init_weights)
        # Depth-aware init for the projections that write into the residual stream.
        # Each block adds two such contributions, so the residual-stream variance would grow ~2L;
        # scaling std by 1/sqrt(2L) keeps it ~constant at init (GPT-2 paper, Sec. 2.3).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.num_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.transformer.gradient_checkpointing = enabled

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kvs: list[KVCache | None] | None = None,
        use_cache: bool = False,
        last_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KVCache] | None]:
        """Return (logits, loss, presents).

        idx, targets: (B, T) int64.  logits: (B, T, V) (or (B, 1, V) if ``last_only``).
        ``targets[:, t]`` must be the token that *follows* ``idx[:, t]`` (the dataset provides the shift).
        """
        pos_offset = 0
        if past_kvs is not None and past_kvs[0] is not None:
            pos_offset = past_kvs[0][0].size(2)
        x = self.embeddings(idx, pos_offset=pos_offset)  # (B, T, C)
        x, presents = self.transformer(x, past_kvs=past_kvs, use_cache=use_cache, pos_offset=pos_offset)
        if last_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)  # (B, T, V): unnormalised log-probabilities
        loss = None
        if targets is not None:
            # Cross-entropy in >= float32 for numerical stability under autocast (fp16/bf16 logits are upcast;
            # float32/float64 logits are left untouched so float64 gradient checking is exact).
            flat = logits.reshape(-1, logits.size(-1))
            if flat.dtype in (torch.float16, torch.bfloat16):
                flat = flat.float()
            loss = F.cross_entropy(flat, targets.reshape(-1), ignore_index=-100)
        return logits, loss, presents

    def num_parameters(self, exclude_position: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_position and self.embeddings.pos_emb is not None:
            n -= self.embeddings.pos_emb.weight.numel()
        return n

    def configure_param_groups(self, weight_decay: float) -> list[dict]:
        """Split parameters: >=2-D tensors (matmul weights, embeddings) are decayed;
        1-D tensors (biases, LayerNorm gain/bias) are not.

        Rationale: weight decay acts as an L2-like regulariser on weight matrices. Decaying LayerNorm
        gains pulls them toward 0 (opposite of their identity-init) and decaying biases has no
        regularisation benefit.
        """
        decay, no_decay = [], []
        for _, p in self.named_parameters():  # named_parameters() already de-duplicates tied weights
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
