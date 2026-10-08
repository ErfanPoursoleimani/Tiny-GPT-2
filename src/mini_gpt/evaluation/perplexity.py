"""Loss / perplexity / bits-per-token evaluation.

``PPL = exp(mean token negative log-likelihood in nats)``  (natural log).  ``bits/token = loss / ln 2``.
Perplexity here is per *BPE token* of this tokenizer, so it is NOT comparable to word-level WikiText numbers
(or to other tokenizers).
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch.utils.data import DataLoader

from mini_gpt.model.gpt import GPT


@torch.no_grad()
def evaluate_loader(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    autocast_ctx=None,
) -> dict[str, float]:
    """Token-weighted mean loss over (up to ``max_batches``) batches of the loader."""
    was_training = model.training
    model.eval()
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    total_nll, total_tokens = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            _, loss, _ = model(x, y)
        n = y.numel()
        total_nll += loss.item() * n
        total_tokens += n
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Evaluation loader yielded no batches")
    loss = total_nll / total_tokens
    return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "bits_per_token": loss / math.log(2), "tokens": total_tokens}
