import pytest

from mini_gpt.tokenizer.tokenizer import BPETokenizer
from mini_gpt.tokenizer.train_tokenizer import train_bpe_tokenizer

CORPUS = ["The quick brown fox jumps over the lazy dog. " * 20, "Language models predict the next token. " * 20,
          "Numbers like 12345 and symbols #!? appear too. " * 10]


@pytest.fixture(scope="module")
def tok():
    return train_bpe_tokenizer(texts=CORPUS, vocab_size=400, min_frequency=1)


def test_roundtrip_ascii(tok):
    s = "The quick brown fox predicts the next token."
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode_and_unseen_bytes(tok):
    s = "Café 日本語 🚀 naïve"  # byte-level BPE never needs <unk>
    ids = tok.encode(s)
    assert tok.unk_id not in ids
    assert tok.decode(ids) == s


def test_special_tokens(tok):
    assert {tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id} == {0, 1, 2, 3}
    ids = tok.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "hello"  # specials skipped


def test_vocab_size_bounded(tok):
    assert tok.vocab_size <= 400


def test_save_load(tok, tmp_path):
    tok.save(tmp_path)
    assert (tmp_path / "vocab.json").exists() and (tmp_path / "merges.txt").exists()
    t2 = BPETokenizer.load(tmp_path)
    s = "the lazy dog"
    assert t2.encode(s) == tok.encode(s)
    assert t2.fingerprint() == tok.fingerprint()


def test_missing_tokenizer_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_tokenizer"):
        BPETokenizer.load(tmp_path / "nope")
