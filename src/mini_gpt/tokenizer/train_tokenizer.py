"""Train a byte-level BPE tokenizer on the *training split only* (no validation/test leakage)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers

from mini_gpt.tokenizer.tokenizer import SPECIAL_TOKENS, BPETokenizer


def train_bpe_tokenizer(
    files: list[str | Path] | None = None,
    texts: Iterable[str] | None = None,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> BPETokenizer:
    """Byte-level BPE (GPT-2 style): every byte is in the base alphabet, so *any* string can be
    encoded without <unk>. ``add_prefix_space=False`` matches GPT-2."""
    if (files is None) == (texts is None):
        raise ValueError("Provide exactly one of `files` or `texts`")
    tk = HFTokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    if files is not None:
        tk.train([str(f) for f in files], trainer)
    else:
        tk.train_from_iterator(texts, trainer)
    return BPETokenizer(tk)
