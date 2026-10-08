#!/usr/bin/env python
"""Pipeline sanity check: overfit one tiny sentence. Exercises tokenizer -> dataset -> forward -> loss -> backward ->
optimizer -> checkpoint -> generation. If this fails, do NOT start real training.

    python scripts/overfit.py
"""
import argparse
import json
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from mini_gpt.config import ModelConfig
from mini_gpt.data.dataset import TokenChunkDataset
from mini_gpt.generation.generate import generate_text
from mini_gpt.model.gpt import GPT
from mini_gpt.tokenizer.train_tokenizer import train_bpe_tokenizer
from mini_gpt.training.checkpoint import load_checkpoint, save_checkpoint
from mini_gpt.config import Config
from mini_gpt.utils.seed import set_seed

import numpy as np


def run_overfit(steps: int = 300, lr: float = 3e-3, seed: int = 0, verbose: bool = True) -> dict:
    set_seed(seed)
    text = "the cat sat on the mat . " * 200
    tok = train_bpe_tokenizer(texts=[text], vocab_size=300, min_frequency=1)
    ids = np.array(tok.encode(text), dtype=np.uint16)
    cfg = ModelConfig(vocab_size=tok.vocab_size, context_length=16, embedding_dim=64, num_layers=2, num_heads=2, dropout=0.0)
    model = GPT(cfg)
    ds = TokenChunkDataset(ids, cfg.context_length, stride=3)
    x = torch.stack([ds[i][0] for i in range(min(16, len(ds)))])
    y = torch.stack([ds[i][1] for i in range(min(16, len(ds)))])
    opt = torch.optim.AdamW(model.configure_param_groups(0.0), lr=lr, betas=(0.9, 0.95))
    first = None
    for step in range(steps):
        _, loss, _ = model(x, y)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if first is None:
            first = loss.item()
        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"step {step:4d} loss {loss.item():.5f}")
    final = loss.item()
    # checkpoint round-trip
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "overfit.pt"
        full_cfg = Config(); full_cfg.model = cfg
        save_checkpoint(p, model, opt, None, None, steps, 0.0, final, full_cfg, tok.info())
        ck = load_checkpoint(p)
        m2 = GPT(cfg); m2.load_state_dict(ck["model"]); m2.eval(); model.eval()
        with torch.no_grad():
            diff = (model(x)[0] - m2(x)[0]).abs().max().item()
    prompt = "the cat sat"
    gen = generate_text(model, tok, prompt, torch.device("cpu"), max_new_tokens=12, greedy=True)
    return {"initial_loss": first, "final_loss": final, "checkpoint_max_logit_diff": diff, "generation": gen,
            "passed": final < 0.05 and diff < 1e-6}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    res = run_overfit(args.steps)
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["passed"] else 1)
