"""Checkpoint evaluation (loss/perplexity on train-subset/validation/test) and fixed qualitative prompts."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from mini_gpt.data.dataloader import make_loader
from mini_gpt.data.splits import ensure_tokenized
from mini_gpt.evaluation.perplexity import evaluate_loader
from mini_gpt.generation.generate import generate_text
from mini_gpt.model.gpt import GPT

# Deterministic qualitative evaluation prompt set (greedy decoding => reproducible text).
EVAL_PROMPTS = [
    "Artificial intelligence is",
    "The history of the city began",
    "In 1998, the team",
    "The river flows through",
    "Scientists have discovered that",
    "The first season of the television series",
]


def generate_fixed_samples(model: GPT, tokenizer, device, max_new_tokens: int = 60, autocast_ctx=None) -> list[dict]:
    """Greedy continuations of EVAL_PROMPTS plus one seeded temperature sample each."""
    out = []
    for p in EVAL_PROMPTS:
        greedy = generate_text(model, tokenizer, p, device, max_new_tokens=max_new_tokens, greedy=True,
                               repetition_penalty=1.1, autocast_ctx=autocast_ctx)
        sampled = generate_text(model, tokenizer, p, device, max_new_tokens=max_new_tokens, temperature=0.8,
                                top_k=50, seed=1234, autocast_ctx=autocast_ctx)
        out.append({"prompt": p, "greedy": greedy, "sampled_t0.8_k50": sampled})
    return out


def format_samples(samples: list[dict], step: int | str) -> str:
    lines = [f"=== samples @ step {step} ===", ""]
    for s in samples:
        lines += [f"PROMPT: {s['prompt']}", f"  greedy : {s['greedy']!r}", f"  sampled: {s['sampled_t0.8_k50']!r}", ""]
    return "\n".join(lines)


def evaluate_model_on_splits(model, cfg, tokenizer, device, batch_size: int, autocast_ctx=None,
                             train_batches: int = 50, max_batches: int | None = None) -> dict:
    bins = ensure_tokenized(cfg.data.dataset, cfg.data.processed_dir, tokenizer)
    T = cfg.model.context_length
    results = {}
    for split, mb in (("train_subset", train_batches), ("validation", max_batches), ("test", max_batches)):
        key = "train" if split == "train_subset" else split
        loader = make_loader(bins[key], T, batch_size, shuffle=False, max_tokens=cfg.data.max_eval_tokens if key != "train" else None)
        results[split] = evaluate_loader(model, loader, device, max_batches=mb, autocast_ctx=autocast_ctx)
    return results


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
