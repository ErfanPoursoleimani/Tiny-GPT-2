"""Download, clean and split raw text corpora.

WikiText-2 source: the pre-split train/valid/test files shipped in pytorch/examples
(``word_language_model/data/wikitext-2``). That copy is the *word-level tokenised* variant
(spaces around punctuation, ``@-@`` placeholders, ``<unk>`` for rare words). We detokenise lightly so
the BPE tokenizer sees natural text. Official splits are kept untouched (no leakage).
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

WIKITEXT2_BASE = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

_ARTICLE_RE = re.compile(r"^ = [^=].* = $")  # level-1 heading => new article


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def detokenize_wikitext(text: str) -> str:
    """Undo WikiText's word-level tokenisation artefacts (best-effort, deterministic)."""
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([.,;:!?%)\]}'])", r"\1", text)  # no space before closing punctuation
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    text = text.replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re").replace(" 've", "'ve")
    text = text.replace(" 'd", "'d").replace(" 'll", "'ll").replace(" 'm", "'m")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def clean_text(text: str, detok_wikitext: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if detok_wikitext:
        text = detokenize_wikitext(text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln.strip() if detok_wikitext else ln)
    return "\n".join(out).strip() + "\n"


def split_documents(text: str, dataset: str) -> list[str]:
    """Split a corpus into documents. WikiText: one document per article; others: paragraphs."""
    if dataset.startswith("wikitext"):
        docs: list[list[str]] = []
        for ln in text.split("\n"):
            if _ARTICLE_RE.match(f" {ln} ") or _ARTICLE_RE.match(ln):
                docs.append([ln])
            elif docs:
                docs[-1].append(ln)
            else:
                docs.append([ln])
        return ["\n".join(d).strip() + "\n" for d in docs if "".join(d).strip()]
    return [p.strip() + "\n" for p in re.split(r"\n\s*\n", text) if p.strip()]


def prepare_wikitext2(raw_dir: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, fname in (("train", "train.txt"), ("validation", "valid.txt"), ("test", "test.txt")):
        src = download(f"{WIKITEXT2_BASE}/{fname}", raw_dir / "wikitext2" / fname)
        raw = src.read_text(encoding="utf-8")
        # WikiText marks article headings as " = Title = ". Detokenising turns them into "= Title =".
        cleaned = clean_text(raw, detok_wikitext=True)
        dst = out_dir / f"{split}.txt"
        dst.write_text(cleaned, encoding="utf-8")
        paths[split] = dst
    return paths


def prepare_single_file(src: Path, out_dir: Path, fractions: tuple[float, float, float] = (0.9, 0.05, 0.05)) -> dict[str, Path]:
    """Contiguous (not shuffled) split of one file: first 90% train, next 5% validation, last 5% test.

    Contiguous splitting avoids near-duplicate neighbouring passages ending up in different splits,
    which a random paragraph shuffle would allow (a mild form of leakage).
    Splits are cut at paragraph boundaries.
    """
    text = clean_text(src.read_text(encoding="utf-8"))
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    n = len(paras)
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    parts = {"train": paras[:a], "validation": paras[a:b], "test": paras[b:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, ps in parts.items():
        dst = out_dir / f"{split}.txt"
        dst.write_text("\n\n".join(ps).strip() + "\n", encoding="utf-8")
        paths[split] = dst
    return paths


def prepare_dataset(dataset: str, raw_dir: str | Path, processed_dir: str | Path, custom_path: str | None = None) -> dict[str, Path]:
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    out_dir = processed_dir / dataset
    if dataset == "wikitext2":
        return prepare_wikitext2(raw_dir, out_dir)
    if dataset == "tinyshakespeare":
        src = download(TINY_SHAKESPEARE, raw_dir / "tinyshakespeare" / "input.txt")
        return prepare_single_file(src, out_dir)
    if dataset == "custom":
        if not custom_path:
            raise ValueError("dataset=custom requires data.custom_path (a .txt file)")
        return prepare_single_file(Path(custom_path), out_dir)
    raise ValueError(
        f"Unknown dataset {dataset!r}. Supported out of the box: wikitext2, tinyshakespeare, custom. "
        "WikiText-103 / OpenWebText / TinyStories: export to a .txt and use dataset=custom "
        "(see README)."
    )
