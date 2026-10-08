"""DataLoader factory. Batches stay on the CPU (pinned when CUDA is used) and are moved to the GPU per step."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from mini_gpt.data.dataset import TokenChunkDataset
from mini_gpt.data.splits import load_tokens


def make_loader(
    bin_path,
    context_length: int,
    batch_size: int,
    shuffle: bool,
    max_tokens: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    tokens = load_tokens(bin_path, max_tokens)
    ds = TokenChunkDataset(tokens, context_length)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=g if shuffle else None,
    )
