"""Autoregressive generation with temperature / top-k / top-p / repetition penalty and an optional KV cache.

Pipeline per step:  logits[-1] -> repetition penalty -> temperature -> top-k -> top-p -> softmax -> sample.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_gpt.model.gpt import GPT


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style penalty: for tokens already seen, divide positive logits / multiply negative logits by ``penalty``."""
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    logits = logits.clone()
    seen = torch.unique(generated)
    score = logits[:, seen]
    logits[:, seen] = torch.where(score > 0, score / penalty, score * penalty)
    return logits


def top_k_top_p_filter(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Mask logits outside the top-k set and outside the smallest nucleus with cumulative prob >= top_p."""
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = cum - probs > top_p  # keep the token that crosses the threshold
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    greedy: bool = False,
    use_cache: bool = True,
    seed: int | None = None,
    eos_id: int | None = None,
    autocast_ctx=None,
) -> torch.Tensor:
    """Generate ``max_new_tokens`` tokens after the prompt ``idx`` of shape (B, T0). Returns (B, T0+new).

    * Without cache: the (cropped) full prefix is re-run every step: O(T) work per new token.
    * With cache: only the newest token is fed; keys/values of earlier tokens are reused.
      Learned absolute positions cannot extend beyond ``context_length``; when the window is full we
      crop to the last ``context_length-1`` tokens and rebuild the cache (sliding window).
    """
    was_training = model.training
    model.eval()
    device = idx.device
    T_max = model.cfg.context_length
    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

    import contextlib

    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    past = None
    out = idx
    for _ in range(max_new_tokens):
        if use_cache:
            if past is not None and past[0][0].size(2) >= T_max:  # cache full: slide the window
                past = None
            if past is None:
                inp = out[:, -(T_max - 1) :] if out.size(1) >= T_max else out
            else:
                inp = out[:, -1:]
            with ctx:
                logits, _, past = model(inp, past_kvs=past, use_cache=True, last_only=True)
        else:
            inp = out[:, -T_max:]
            with ctx:
                logits, _, _ = model(inp, last_only=True)
        logits = logits[:, -1, :].float()  # (B, V)
        logits = apply_repetition_penalty(logits, out, repetition_penalty)
        if greedy or temperature == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_top_p_filter(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen)
        out = torch.cat([out, nxt], dim=1)
        if eos_id is not None and out.size(0) == 1 and nxt.item() == eos_id:
            break
    model.train(was_training)
    return out


def generate_text(model: GPT, tokenizer, prompt: str, device: torch.device, **kwargs) -> str:
    ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, **kwargs)
    return tokenizer.decode(out[0].tolist())
