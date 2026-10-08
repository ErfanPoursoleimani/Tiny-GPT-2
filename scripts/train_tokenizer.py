#!/usr/bin/env python
"""Train the byte-level BPE tokenizer on the training split only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_gpt.config import load_config
from mini_gpt.tokenizer.tokenizer import tokenizer_dir_for
from mini_gpt.tokenizer.train_tokenizer import train_bpe_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--vocab-size", type=int, default=None)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dataset = args.dataset or cfg.data.dataset
    vocab = args.vocab_size or cfg.tokenizer.vocab_size
    train_file = Path(cfg.data.processed_dir) / dataset / "train.txt"
    if not train_file.exists():
        sys.exit(f"{train_file} not found. Run scripts/prepare_data.py first.")
    tok = train_bpe_tokenizer(files=[train_file], vocab_size=vocab, min_frequency=cfg.tokenizer.min_frequency)
    out = tokenizer_dir_for(cfg.data.tokenizer_dir, dataset, vocab)
    tok.save(out)
    print(f"Saved tokenizer to {out}  (vocab_size={tok.vocab_size}, requested={vocab})")
    sample = "Artificial intelligence is changing the world."
    ids = tok.encode(sample)
    print(f"{sample!r} -> {ids}\n  decoded: {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()
