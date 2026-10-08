#!/usr/bin/env python
"""Benchmark forward/backward latency, throughput, memory, attention implementations and KV-cache generation.

    python scripts/benchmark.py --config configs/small.yaml
    python scripts/benchmark.py --gradient-check            # finite-difference check on a tiny float64 model
"""
import argparse
import copy
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from mini_gpt.config import load_config
from mini_gpt.evaluation.benchmarks import (benchmark_generation, benchmark_train_step, gradient_check,
                                            kv_cache_equivalence)
from mini_gpt.utils.device import detect_precision, get_device, hardware_summary
from mini_gpt.utils.parameter_count import count_parameters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gradient-check", action="store_true", help="only run the finite-difference gradient check")
    ap.add_argument("--vocab-size", type=int, default=None, help="override model vocab (benchmarks don't need a tokenizer)")
    ap.add_argument("--out", default="outputs/metrics/benchmark.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.vocab_size:
        cfg.model.vocab_size = args.vocab_size
    device = get_device(args.device)

    if args.gradient_check:
        tiny = copy.deepcopy(cfg.model)
        tiny.vocab_size, tiny.context_length, tiny.embedding_dim, tiny.num_layers, tiny.num_heads = 50, 8, 16, 2, 2
        tiny.attention_impl = "manual"
        res = gradient_check(tiny, n_checks=12)
        for r in res:
            print(f"{r['param']:<40} analytic={r['analytic']:+.6e} numeric={r['numeric']:+.6e} abs_err={r['abs_error']:.1e} "
                  f"rel_err={r['rel_error']:.1e} {'ok' if r['passed'] else 'FAIL'}")
        ok = all(r["passed"] for r in res)
        print(f"\nfinite-difference gradient check (float64, eps=1e-5): {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    bs = args.batch_size or cfg.training.batch_size
    prec = detect_precision(device, True, cfg.precision.dtype)
    hw = hardware_summary(device)
    results: dict = {"hardware": hw, "batch_size": bs, "context_length": cfg.model.context_length,
                     "parameters": count_parameters(__import__("mini_gpt.model.gpt", fromlist=["GPT"]).GPT(cfg.model))}
    print(json.dumps(hw, indent=2))

    # CPU "mixed" = bf16 autocast on CPU (informational; the training loop itself uses fp32 on CPU).
    mixed_dtype = prec.amp_dtype if prec.mixed_precision else (torch.bfloat16 if device.type == "cpu" else None)
    mixed_label = f"mixed({str(mixed_dtype).replace('torch.', '')})" if mixed_dtype else "mixed(n/a)"
    rows = []
    for attn in ("manual", "sdpa"):
        for label, dt in (("fp32", None), (mixed_label, mixed_dtype)):
            if label == "mixed(n/a)":
                continue
            c = copy.deepcopy(cfg.model); c.attention_impl = attn
            r = benchmark_train_step(c, bs, device, dt, steps=args.steps)
            r.update(attention=attn, precision=label)
            rows.append(r)
            print(f"[train] attn={attn:<6} prec={label:<16} fwd {r['forward_ms']:.0f} ms | bwd {r['backward_ms']:.0f} ms | "
                  f"{r['tokens_per_sec']:.0f} tok/s | peak VRAM {r['peak_vram_mb']:.0f} MB | RSS {r['cpu_rss_mb']:.0f} MB")
    results["train_step"] = rows

    gens = []
    for cache in (False, True):
        c = copy.deepcopy(cfg.model); c.attention_impl = "sdpa"
        g = benchmark_generation(c, device, prompt_len=32, new_tokens=args.gen_tokens, use_cache=cache)
        g["kv_cache"] = cache
        gens.append(g)
        print(f"[generate] kv_cache={cache!s:<5} {g['tokens_per_sec']:.1f} tokens/s ({g['seconds']:.2f}s for {args.gen_tokens} tokens)")
    results["generation"] = gens
    results["kv_cache_speedup"] = gens[1]["tokens_per_sec"] / gens[0]["tokens_per_sec"]
    c = copy.deepcopy(cfg.model); c.dropout = 0.0
    results["kv_cache_max_abs_logit_diff"] = kv_cache_equivalence(c, device)
    print(f"[kv-cache] speedup x{results['kv_cache_speedup']:.2f} | max |logit diff| vs full forward: {results['kv_cache_max_abs_logit_diff']:.2e}")

    # loss / output equivalence between attention implementations (same weights)
    from mini_gpt.model.gpt import GPT
    torch.manual_seed(0)
    cm = copy.deepcopy(cfg.model); cm.dropout = 0.0; cm.attention_impl = "manual"
    m1 = GPT(cm).to(device).eval()
    cs = copy.deepcopy(cm); cs.attention_impl = "sdpa"
    m2 = GPT(cs).to(device).eval(); m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, cm.vocab_size, (2, cm.context_length), device=device)
    with torch.no_grad():
        l1, loss1, _ = m1(x, x); l2, loss2, _ = m2(x, x)
    results["manual_vs_sdpa"] = {"max_abs_logit_diff": (l1 - l2).abs().max().item(), "loss_manual": loss1.item(), "loss_sdpa": loss2.item()}
    print(f"[equivalence] manual vs sdpa: max |logit diff| {results['manual_vs_sdpa']['max_abs_logit_diff']:.2e}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
