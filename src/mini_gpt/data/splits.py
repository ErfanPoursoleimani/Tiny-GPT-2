"""Tokenise split files into flat uint16 token streams cached on disk (``.bin``)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mini_gpt.data.preprocessing import split_documents
from mini_gpt.tokenizer.tokenizer import BPETokenizer

SPLITS = ("train", "validation", "test")


def tokenize_split(text_path: Path, tokenizer: BPETokenizer, dataset: str) -> np.ndarray:
    """Encode each document and append <eos> after it, then concatenate into one token stream."""
    text = text_path.read_text(encoding="utf-8")
    docs = split_documents(text, dataset)
    stream: list[int] = []
    # encode in batches to bound memory
    for i in range(0, len(docs), 512):
        for ids in tokenizer.encode_batch(docs[i : i + 512]):
            stream.extend(ids)
            stream.append(tokenizer.eos_id)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("vocab too large for uint16 storage")
    return np.asarray(stream, dtype=np.uint16)


def ensure_tokenized(dataset: str, processed_dir: str | Path, tokenizer: BPETokenizer, force: bool = False) -> dict[str, Path]:
    """Create ``<split>.bin`` files if missing or if the tokenizer changed (fingerprint mismatch)."""
    d = Path(processed_dir) / dataset
    fp = tokenizer.fingerprint()
    cache = d / f"tok_{fp}"  # one cache per tokenizer, so different vocab sizes never clobber each other
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "tokenized_meta.json"
    paths = {s: cache / f"{s}.bin" for s in SPLITS}
    fresh = (
        not force
        and meta_path.exists()
        and json.loads(meta_path.read_text()).get("fingerprint") == fp
        and all(p.exists() for p in paths.values())
    )
    if fresh:
        return paths
    meta = {"fingerprint": fp, "vocab_size": tokenizer.vocab_size, "tokens": {}}
    for s in SPLITS:
        txt = d / f"{s}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"{txt} missing. Run scripts/prepare_data.py first.")
        arr = tokenize_split(txt, tokenizer, dataset)
        arr.tofile(paths[s])
        meta["tokens"][s] = int(arr.size)
    meta_path.write_text(json.dumps(meta, indent=2))
    return paths


def load_tokens(path: str | Path, max_tokens: int | None = None) -> np.ndarray:
    arr = np.memmap(path, dtype=np.uint16, mode="r")  # stays on disk / in page cache, never on the GPU
    return arr[:max_tokens] if max_tokens else arr
