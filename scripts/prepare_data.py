#!/usr/bin/env python
"""Download/clean/split a dataset into data/processed/<dataset>/{train,validation,test}.txt.

Tokenisation to .bin happens after the tokenizer is trained (train_tokenizer.py) or automatically at training time.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_gpt.config import load_config
from mini_gpt.data.preprocessing import prepare_dataset
from mini_gpt.data.splits import ensure_tokenized
from mini_gpt.tokenizer.tokenizer import BPETokenizer, tokenizer_dir_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--dataset", default=None, help="wikitext2 | tinyshakespeare | custom (overrides config)")
    ap.add_argument("--custom-path", default=None)
    ap.add_argument("--tokenize", action="store_true", help="also tokenise splits (requires a trained tokenizer)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dataset = args.dataset or cfg.data.dataset
    paths = prepare_dataset(dataset, cfg.data.raw_dir, cfg.data.processed_dir, args.custom_path or cfg.data.custom_path)
    for split, p in paths.items():
        print(f"{split:<10} {p}  ({p.stat().st_size / 1e6:.2f} MB)")
    if args.tokenize:
        tok = BPETokenizer.load(tokenizer_dir_for(cfg.data.tokenizer_dir, dataset, cfg.tokenizer.vocab_size))
        bins = ensure_tokenized(dataset, cfg.data.processed_dir, tok, force=True)
        for s, p in bins.items():
            print(f"{s:<10} {p}  ({p.stat().st_size // 2:,} tokens)")


if __name__ == "__main__":
    main()
