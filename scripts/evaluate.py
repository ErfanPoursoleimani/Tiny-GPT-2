#!/usr/bin/env python
"""Evaluate a checkpoint: loss / perplexity / bits-per-token on (a train subset,) validation and test."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from mini_gpt.evaluation.evaluation import evaluate_model_on_splits, save_metrics
from mini_gpt.tokenizer.tokenizer import BPETokenizer, tokenizer_dir_for
from mini_gpt.training.checkpoint import model_from_checkpoint
from mini_gpt.training.trainer import make_autocast
from mini_gpt.utils.device import detect_precision, get_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-batches", type=int, default=None, help="limit eval batches (default: full splits)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    model, cfg, ckpt = model_from_checkpoint(args.checkpoint, device)
    prec = detect_precision(device, cfg.precision.mixed_precision, cfg.precision.dtype)
    tok = BPETokenizer.load(tokenizer_dir_for(cfg.data.tokenizer_dir, cfg.data.dataset, cfg.tokenizer.vocab_size))
    res = evaluate_model_on_splits(model, cfg, tok, device, args.batch_size, autocast_ctx=make_autocast(prec), max_batches=args.max_batches)
    res["checkpoint"], res["step"] = args.checkpoint, ckpt["step"]
    out = args.out or f"outputs/metrics/eval_{Path(args.checkpoint).stem}.json"
    save_metrics(res, out)
    print(json.dumps(res, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
