"""Speed / memory benchmarks and finite-difference gradient checking."""
from __future__ import annotations

import contextlib
import copy
import time

import torch

from mini_gpt.config import ModelConfig
from mini_gpt.generation.generate import generate
from mini_gpt.model.gpt import GPT
from mini_gpt.utils.profiling import memory_stats, reset_peak, sync


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def benchmark_train_step(
    cfg: ModelConfig, batch_size: int, device: torch.device, amp_dtype: torch.dtype | None = None,
    steps: int = 5, warmup: int = 2,
) -> dict[str, float]:
    """Time forward and backward separately (synchronised) and report tokens/sec + peak memory."""
    model = GPT(cfg).to(device).train()
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.context_length), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.context_length), device=device)
    reset_peak(device)
    fwd, bwd = [], []
    scaler_free = amp_dtype != torch.float16  # fp16 would need GradScaler; benchmark only measures timing
    for i in range(warmup + steps):
        model.zero_grad(set_to_none=True)
        sync(device); t0 = time.perf_counter()
        with _autocast(device, amp_dtype):
            _, loss, _ = model(x, y)
        sync(device); t1 = time.perf_counter()
        loss.backward()
        sync(device); t2 = time.perf_counter()
        if i >= warmup:
            fwd.append(t1 - t0); bwd.append(t2 - t1)
    f, b = sum(fwd) / len(fwd), sum(bwd) / len(bwd)
    mem = memory_stats(device)
    return {
        "forward_ms": f * 1000, "backward_ms": b * 1000,
        "tokens_per_sec": batch_size * cfg.context_length / (f + b),
        "peak_vram_mb": mem["peak_mb"], "cpu_rss_mb": mem["cpu_rss_mb"], "loss": float(loss.item()),
        "num_params": float(sum(p.numel() for p in model.parameters())), "scaler_free": float(scaler_free),
    }


def benchmark_generation(
    cfg: ModelConfig, device: torch.device, prompt_len: int = 32, new_tokens: int = 64, use_cache: bool = True,
    amp_dtype: torch.dtype | None = None, repeats: int = 2,
) -> dict[str, float]:
    model = GPT(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    ctx = _autocast(device, amp_dtype)
    generate(model, prompt, 4, greedy=True, use_cache=use_cache, autocast_ctx=ctx)  # warmup
    times = []
    for _ in range(repeats):
        sync(device); t0 = time.perf_counter()
        generate(model, prompt, new_tokens, greedy=True, use_cache=use_cache, autocast_ctx=_autocast(device, amp_dtype))
        sync(device); times.append(time.perf_counter() - t0)
    t = min(times)
    return {"seconds": t, "tokens_per_sec": new_tokens / t}


def kv_cache_equivalence(cfg: ModelConfig, device: torch.device, n: int = 24) -> float:
    """Max |logit difference| between cached incremental decoding and a full forward pass (should be ~1e-5)."""
    torch.manual_seed(0)
    model = GPT(cfg).to(device).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, n), device=device)
    with torch.no_grad():
        full, _, _ = model(idx)
        past = None
        outs = []
        for t in range(n):
            lg, _, past = model(idx[:, t : t + 1], past_kvs=past, use_cache=True)
            outs.append(lg)
        inc = torch.cat(outs, dim=1)
    return (full - inc).abs().max().item()


def gradient_check(cfg: ModelConfig, n_checks: int = 8, eps: float = 1e-5, seed: int = 0,
                   atol: float = 1e-8, rtol: float = 1e-4) -> list[dict]:
    """Compare autograd gradients with central finite differences in float64 on randomly chosen scalars.

    Uses a tiny model with dropout disabled. Not part of normal training (cost: 2 forward passes per scalar).
    """
    cfg = copy.deepcopy(cfg)
    cfg.dropout = 0.0
    torch.manual_seed(seed)
    model = GPT(cfg).double().eval()
    x = torch.randint(0, cfg.vocab_size, (2, min(8, cfg.context_length)))
    y = torch.randint(0, cfg.vocab_size, x.shape)
    model.zero_grad()
    _, loss, _ = model(x, y)
    loss.backward()
    rng = torch.Generator().manual_seed(seed)
    results = []
    names = [n for n, p in model.named_parameters() if p.grad is not None and p.numel() > 1]
    for _ in range(n_checks):
        name = names[int(torch.randint(len(names), (1,), generator=rng))]
        p = dict(model.named_parameters())[name]
        flat_i = int(torch.randint(p.numel(), (1,), generator=rng))
        idx = torch.unravel_index(torch.tensor(flat_i), p.shape)
        analytic = p.grad[idx].item()
        with torch.no_grad():
            orig = p[idx].item()
            p[idx] = orig + eps; lp = model(x, y)[1].item()
            p[idx] = orig - eps; lm = model(x, y)[1].item()
            p[idx] = orig
        numeric = (lp - lm) / (2 * eps)
        abs_err = abs(analytic - numeric)
        ref = max(abs(analytic), abs(numeric))
        # Central differences in float64 have a roundoff floor of ~1e-16 * |loss| / eps ~ 1e-10, so tiny gradients
        # are judged with an absolute tolerance (like torch.allclose: atol + rtol * |ref|).
        results.append({"param": name, "analytic": analytic, "numeric": numeric, "abs_error": abs_err,
                        "rel_error": abs_err / max(ref, 1e-12), "passed": abs_err <= atol + rtol * ref})
    return results
