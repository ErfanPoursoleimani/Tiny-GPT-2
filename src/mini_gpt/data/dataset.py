"""Fixed-length causal-LM chunks over a flat token stream.

For a chunk starting at ``s``::

    x = tokens[s     : s + T]        # inputs
    y = tokens[s + 1 : s + T + 1]    # targets: the *next* token at every position

so position ``t`` is trained to predict ``tokens[s+t+1]`` from ``tokens[s : s+t+1]`` only.
(The causal mask inside the model guarantees position t never sees tokens > t.)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenChunkDataset(Dataset):
    def __init__(self, tokens: np.ndarray, context_length: int, stride: int | None = None):
        if len(tokens) < context_length + 1:
            raise ValueError(f"Token stream ({len(tokens)}) shorter than context_length+1 ({context_length + 1})")
        self.tokens = tokens
        self.T = context_length
        self.stride = stride or context_length  # non-overlapping by default
        self.n = (len(tokens) - context_length - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = i * self.stride
        chunk = np.asarray(self.tokens[s : s + self.T + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
