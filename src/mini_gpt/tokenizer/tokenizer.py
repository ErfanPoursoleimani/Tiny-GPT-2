"""Thin wrapper around a Hugging Face ``tokenizers`` byte-level BPE model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOKENIZER_FILE = "tokenizer.json"


class BPETokenizer:
    """Encode/decode text <-> token ids. Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3."""

    def __init__(self, hf_tokenizer: HFTokenizer, path: str | Path | None = None):
        self.tk = hf_tokenizer
        self.path = str(path) if path else None
        self.pad_id = self.tk.token_to_id("<pad>")
        self.unk_id = self.tk.token_to_id("<unk>")
        self.bos_id = self.tk.token_to_id("<bos>")
        self.eos_id = self.tk.token_to_id("<eos>")

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tk.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tk.encode_batch(texts)]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / TOKENIZER_FILE
        self.tk.save(str(path))
        # Also export the classic GPT-2 style vocab.json + merges.txt for inspection / interoperability.
        data = json.loads(path.read_text(encoding="utf-8"))
        model = data["model"]
        (directory / "vocab.json").write_text(json.dumps(model["vocab"], ensure_ascii=False), encoding="utf-8")
        merges = ["#version: 0.2"] + [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
        (directory / "merges.txt").write_text("\n".join(merges) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        path = Path(directory)
        if path.is_dir():
            path = path / TOKENIZER_FILE
        if not path.exists():
            raise FileNotFoundError(f"No tokenizer at {path}. Run scripts/train_tokenizer.py first.")
        return cls(HFTokenizer.from_file(str(path)), path)

    def fingerprint(self) -> str:
        """Hash of the tokenizer definition; used to detect stale tokenised caches."""
        return hashlib.sha1(self.tk.to_str().encode("utf-8")).hexdigest()[:16]

    def info(self) -> dict:
        return {
            "type": "byte-level-bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": {"pad": self.pad_id, "unk": self.unk_id, "bos": self.bos_id, "eos": self.eos_id},
            "fingerprint": self.fingerprint(),
            "path": self.path,
        }


def tokenizer_dir_for(base_dir: str | Path, dataset: str, vocab_size: int) -> Path:
    """Canonical location: <tokenizer_dir>/<dataset>_v<vocab_size>."""
    return Path(base_dir) / f"{dataset}_v{vocab_size}"
