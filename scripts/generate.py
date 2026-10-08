#!/usr/bin/env python
"""Generate text from a checkpoint."""
import argparse

import _bootstrap  # noqa: F401
import torch

from mini_gpt.generation.generate import generate
from mini_gpt.tokenizer.tokenizer import BPETokenizer, tokenizer_dir_for
from mini_gpt.training.checkpoint import model_from_checkpoint
from mini_gpt.training.trainer import make_autocast
from mini_gpt.utils.device import detect_precision, get_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", default="Artificial intelligence is")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="disable the KV cache")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()
    device = get_device(args.device)
    model, cfg, _ = model_from_checkpoint(args.checkpoint, device)
    tok = BPETokenizer.load(tokenizer_dir_for(cfg.data.tokenizer_dir, cfg.data.dataset, cfg.tokenizer.vocab_size))
    prec = detect_precision(device, cfg.precision.mixed_precision, cfg.precision.dtype)
    ids = tok.encode(args.prompt, add_bos=True)
    for i in range(args.num_samples):
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(model, idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                       repetition_penalty=args.repetition_penalty, greedy=args.greedy, use_cache=not args.no_cache,
                       seed=None if args.seed is None else args.seed + i, autocast_ctx=make_autocast(prec))
        print(f"--- sample {i + 1} ---\n{tok.decode(out[0].tolist())}\n")


if __name__ == "__main__":
    main()
