# Tokenizer

We train a **byte-level BPE** (GPT-2 style) with the `tokenizers` library on the **training split only**.

* Start from 256 byte symbols; repeatedly merge the most frequent adjacent pair until `vocab_size` is reached. Every
  string is representable, so there is no real out-of-vocabulary case (`<unk>` exists only as a placeholder; the test
  suite asserts emoji/CJK round-trip without it). Pre-tokenisation uses the GPT-2 regex-free ByteLevel splitter with
  `add_prefix_space=False`, so a leading space is part of the following word ("Ġworld").
* Special tokens: `<pad>=0, <unk>=1, <bos>=2, <eos>=3`. `<eos>` is appended after every document in the training
  stream so the model learns document boundaries; `<bos>` is prepended to prompts at generation time. `<pad>` is
  reserved (we pack fixed-length chunks, so padding is not used and no loss masking is needed).
* Artifacts: `tokenizer.json` (authoritative), plus GPT-2-compatible `vocab.json` and `merges.txt`.
* Directory: `data/tokenizer/<dataset>_v<vocab_size>/` so different vocab sizes never overwrite each other.
  Tokenised caches are keyed by a SHA-1 fingerprint of the tokenizer.

## Why autoregressive subword tokenisation (and not BERT-style)

A causal LM must be able to *generate arbitrary text* one token at a time and must give a proper probability to every
possible continuation. Byte-level BPE guarantees coverage (no `[UNK]` holes), keeps sequences short compared with
characters, and reversibly round-trips whitespace/case/punctuation because spaces are part of tokens. BERT's WordPiece
setup is designed for a *bidirectional encoder*: it lowercases/strips accents optionally, normalises whitespace, uses
`##` continuation pieces and `[CLS]/[SEP]/[MASK]` tokens for masked-language-modelling and sentence-pair tasks. Those
normalisations are lossy (decode ≠ original) and the special tokens serve an objective we do not use. GPT-style models
instead predict the next token left-to-right, so we need a lossless, decodable vocabulary and only BOS/EOS-type markers.

## Vocabulary size trade-off

Bigger vocab → shorter sequences (more text per context window) but a larger embedding/softmax (`V·C` parameters and a
`B·T·V` logits tensor, ~500 MB in fp32 at B=8, T=256, V=32k) and rarer tokens that are poorly trained on a small corpus.
On 2–3 M training tokens, 32k is generous; `experiments.md` proposes a vocab-size sweep (8k/16k/32k).

## Caveat: `<unk>` strings in WikiText

WikiText replaces rare words with the literal text `<unk>`. Because `<unk>` is a registered special token, the
tokenizer maps that literal text to id 1. The model therefore learns to emit `<unk>` where WikiText has rare words;
decoding with `skip_special_tokens=True` drops it. This is a property of the dataset, not a tokenizer failure.
