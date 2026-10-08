import numpy as np
import torch

from mini_gpt.data.dataset import TokenChunkDataset
from mini_gpt.data.preprocessing import clean_text, prepare_single_file, split_documents


def test_targets_are_inputs_shifted_by_one():
    tokens = np.arange(100, dtype=np.uint16)
    ds = TokenChunkDataset(tokens, context_length=10)
    x, y = ds[3]
    assert x.tolist() == list(range(30, 40))
    assert y.tolist() == list(range(31, 41))
    assert torch.equal(x[1:], y[:-1])
    assert x.dtype == torch.int64


def test_chunk_count_and_bounds():
    tokens = np.arange(105, dtype=np.uint16)
    ds = TokenChunkDataset(tokens, 10)
    assert len(ds) == (105 - 10 - 1) // 10 + 1
    x, y = ds[len(ds) - 1]
    assert y.max().item() <= 104  # never reads past the stream


def test_too_short_stream_raises():
    import pytest
    with pytest.raises(ValueError):
        TokenChunkDataset(np.arange(5, dtype=np.uint16), 10)


def test_single_file_split_has_no_leakage(tmp_path):
    paras = [f"paragraph number {i} with some unique words w{i}" for i in range(200)]
    src = tmp_path / "corpus.txt"
    src.write_text("\n\n".join(paras))
    paths = prepare_single_file(src, tmp_path / "out")
    sets = {k: set(p.read_text().strip().split("\n\n")) for k, p in paths.items()}
    assert sets["train"].isdisjoint(sets["validation"]) and sets["train"].isdisjoint(sets["test"]) and sets["validation"].isdisjoint(sets["test"])
    assert len(sets["train"]) == 180 and len(sets["validation"]) == 10 and len(sets["test"]) == 10
    # contiguous: validation comes after train in the original order
    assert paras[180] in sets["validation"] and paras[190] in sets["test"]


def test_wikitext_article_split():
    text = "= Alpha =\n\nbody a1\nbody a2\n\n= = Section = =\n\nmore a\n= Beta =\n\nbody b\n"
    docs = split_documents(clean_text(text), "wikitext2")
    assert len(docs) == 2 and "Section" in docs[0] and docs[1].startswith("= Beta =")


def test_ensure_tokenized_cache(tmp_path):
    from mini_gpt.data.splits import ensure_tokenized
    from mini_gpt.tokenizer.train_tokenizer import train_bpe_tokenizer

    d = tmp_path / "ds"; d.mkdir()
    for s in ("train", "validation", "test"):
        (d / f"{s}.txt").write_text("hello world. " * 100)
    tok = train_bpe_tokenizer(texts=["hello world. " * 50], vocab_size=300, min_frequency=1)
    paths = ensure_tokenized("ds", tmp_path, tok)
    assert all(p.exists() for p in paths.values())
    mtime = paths["train"].stat().st_mtime_ns
    paths2 = ensure_tokenized("ds", tmp_path, tok)
    assert paths2["train"].stat().st_mtime_ns == mtime  # cached, not recomputed
    arr = np.fromfile(paths["train"], dtype=np.uint16)
    assert (arr == tok.eos_id).sum() >= 1
