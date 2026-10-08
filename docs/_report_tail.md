
## Not executed in this environment (and the exact commands to run them)

The development sandbox had **no GPU, 1 CPU core and 3 GB RAM**, and no access to huggingface.co. Consequently:

| Item | Status | Command to run it on the RTX 3050 |
|---|---|---|
| CUDA training, BF16/FP16 autocast, GradScaler, fused AdamW | code written, **not executed** (CPU fallback path was) | `python scripts/train.py --config configs/small.yaml --auto-batch-size` |
| VRAM allocated/reserved/peak reporting; `--auto-batch-size` OOM probing | code written, **not executed** | same command; look for `vram alloc/res/peak` in the log |
| GPU benchmark (FP32 vs mixed, manual vs SDPA, KV cache) | **not executed on GPU**; CPU numbers above | `python scripts/benchmark.py --config configs/small.yaml --batch-size 8` |
| Full 10,000-step run (~164 M tokens) | **not executed** (≈ 85 h at the 530 tok/s measured here) | `python scripts/train.py --config configs/small.yaml` |
| Experiments A–D and the research agenda | specified only (`docs/experiments.md`, `configs/experiments/`) | `python scripts/train.py --config configs/experiments/<name>.yaml --checkpoint-dir checkpoints/<name>` |
| Gradient-checkpointing speed/VRAM trade-off | correctness tested (identical gradients); trade-off **not measured** | `python scripts/train.py --config configs/small.yaml --override training.gradient_checkpointing=true` and compare |
| WikiText-103 / OpenWebText / TinyStories | not downloaded (no HF access; larger than needed for the first run) | export to `.txt`, set `data.dataset: custom`, `data.custom_path: ...` |
| `datasets` library | listed in requirements, but unused by the default pipeline (WikiText-2 is fetched from a GitHub raw URL) | — |

## Limitations

* The main run is **200 optimizer steps (409,600 tokens, 0.18 epochs)** on CPU: it demonstrates that training works end to end and
  that loss falls, not that the model is any good. Generated text from such a run is mostly function-word soup; that is expected.
* Validation loss during training uses the first `eval_iters` (16) micro-batches (4,096 tokens) – a noisy but consistent prefix;
  the final numbers use the full splits.
* Perplexity is per BPE token (vocab 32k) and not comparable with word-level WikiText results.
* WikiText-2 as fetched is the word-level-tokenised distribution; our detokeniser is heuristic (e.g. quotation marks keep WikiText spacing).
* CPU timings (1 core) say nothing about GPU speed; the KV-cache speed-up and manual-vs-SDPA ordering may differ on CUDA.
* Seeds make CPU runs repeatable (the tiny smoke run reproduced identical validation losses twice); GPU runs will not be bit-identical.
* Estimated FLOPs and memory figures are model-based approximations.

## Best next research experiment

**Learned vs RoPE positions with a context-length sweep, on a larger corpus.** Hypothesis: RoPE gives equal or lower validation loss than
learned positions at the training length and degrades more gracefully when evaluated at 2× the training length. Independent variable:
`position_encoding.type ∈ {learned, rope}` (× train context 256, eval context 256/512). Dependent: validation bits/token at each eval length.
Controlled: parameters (RoPE removes only 98 k), tokens/update (16,384), LR 3e-4, seed ×3, same data order. Setup: run
`--override position_encoding.type=rope`; both are already implemented and unit-tested (RoPE relative-position property, cache correctness,
float64 gradient check). Expected: small gain at 256, clearer gain at 512. Evaluation: paired comparison across seeds.
Full specifications for all ten prioritised experiments are in `docs/experiments.md`.
