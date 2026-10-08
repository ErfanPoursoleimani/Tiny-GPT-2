# Experiments

All experiments share: dataset WikiText-2 (or a larger `custom` corpus for scaling), the same tokenizer unless it *is*
the variable, seed 42 (repeat with ≥3 seeds before trusting differences smaller than seed noise), identical effective
batch (64 sequences·tokens/update held fixed unless noted), identical LR schedule shape, and the same eval protocol
(`eval_iters`, full val/test at the end). Each run gets its own `--run-name` and `--checkpoint-dir`.

```bash
python scripts/train.py --config configs/experiments/baseline.yaml --checkpoint-dir checkpoints/baseline
```

> **None of the experiments below were run to completion in the development sandbox (CPU only).** They are
> specifications; see `outputs/research_report.md` for what was actually measured.

## A. Model size (`configs/experiments/model_size.yaml`)

| | Tiny | Small | Medium |
|---|---|---|---|
| layers / dim / heads | 4 / 256 / 4 | 6 / 384 / 6 | 8 / 512 / 8 |
| params (tied, V=32k, T=256) | 11.42 M | 23.0 M | 41.74 M |

*Hypothesis*: at fixed data/steps, validation loss decreases monotonically with size until the model starts to overfit
the 2 M-token corpus; on WikiText-2 the gain from Small→Medium will be smaller than Tiny→Small. *Independent*: layers, dim,
heads. *Dependent*: params, peak VRAM, tokens/sec, best val loss/PPL, train–val gap. *Controlled*: tokens/update (keep
64 seq via accumulation), LR (3e-4; larger models may prefer lower), steps, seed. *Expected*: Medium costs roughly 1.5–2× more compute per token
(FLOP-based; measure it); val loss improves but the train–val gap widens. *Evaluation*: report best-val loss per size, plot loss vs
params; check gap.

## B. Context length (`context_length.yaml`; run 128 / 256 / 512)

*Hypothesis*: longer context helps only if the data contains useful long-range structure; WikiText articles are long
enough that 512 should help slightly, with diminishing returns vs. the quadratic cost. *Independent*: `context_length`.
*Dependent*: peak VRAM, tokens/sec, val loss **per token on the same text** (note: windows differ, so also evaluate all
models with the same 512-token windows / sliding evaluation). *Controlled*: tokens/update fixed (adjust batch size inversely
— 16k tokens), model, LR. *Expected*: VRAM ↑ and speed ↓ with `manual` attention markedly (T²), mildly with SDPA; val
loss improvement ≤ a few %. Don't assume longer is better — the extra positions in the learned table are also more parameters to fit.

## C. Learning rate (`learning_rate.yaml`; run 1e-4, 3e-4, 6e-4)

*Hypothesis*: 3e-4 ≈ near-optimal for this size/batch; 1e-4 under-trains in 10k steps; 6e-4 trains faster early and may be
unstable or end similar. *Independent*: peak LR (min LR = 10 % of peak). *Dependent*: loss curves, grad-norm spikes, final val
loss. *Controlled*: everything else. *Expected*: U-shaped final loss vs LR. *Evaluation*: overlay curves from
`outputs/plots/<run>/`; compare best val loss; look for divergence.

## D. Attention implementation (`attention_impl.yaml`; manual vs sdpa)

*Hypothesis*: identical loss curves (same math) with SDPA faster and lower-memory. *Independent*: `attention.implementation`.
*Dependent*: tokens/sec, peak VRAM, loss difference. *Controlled*: seed, data order, everything else. *Expected*: curves match to
within numerical noise (the unit tests show ≤1e-5 logit differences in fp32; bf16 will diverge slightly over time due to
different reduction orders). Quick check: `python scripts/benchmark.py`.

## Prioritised research agenda

1. **Scaling laws** — H: loss follows `L(N) ≈ a·N^{-α} + c` at fixed data. IV: N (6–8 sizes by width/depth), and separately tokens D (steps ×
   tokens/update) using a bigger corpus (WikiText-103 as `custom`). DV: final val loss. Controlled: architecture family, LR tuned per
   size, data. Expect: smooth power law until data-limited. Eval: log–log fit; report compute (`6ND`).
2. **RoPE vs learned positions** (`position_encoding.type`) — H: RoPE ≤ learned loss at equal params, better when evaluating at lengths
   near/above training length. IV: position type. DV: val loss at T, 0.5T; sampling quality. Ctrl: everything else (RoPE removes 98 k params).
3. **Context-length scaling** — see B, extended to 1024 with SDPA + gradient checkpointing.
4. **Vocabulary size** — H: 8k→32k lowers tokens-per-text, so compare **bits per character/byte** (not PPL!) to be fair. IV: vocab {8k,16k,32k}.
   DV: bits/byte, tokens/sec. Ctrl: model dims, text.
5. **Dataset quality** — raw vs cleaned vs near-dedup corpus (e.g. MinHash dedup): H: dedup reduces train–val gap and memorisation. IV: corpus
   version. DV: val loss on a *clean* held-out set, exact-match memorisation rate on prompts.
6. **Learning-rate scaling with batch** — H: optimal LR grows ~√(batch) in this regime. IV: effective batch {32,64,128} × LR grid.
7. **Depth vs width at fixed params** — IV: (L,C) pairs at ≈23 M. DV: val loss, speed, VRAM. Expect: modest effect; deeper slower per step.
8. **Attention efficiency** — see D; extend to T∈{256,512,1024}. DV: tokens/sec, peak VRAM (on the actual GPU).
9. **Gradient checkpointing trade-off** — IV: on/off, at the largest micro-batch that fits. DV: VRAM, tokens/sec; expect VRAM ↓ substantially, time ↑.
10. **Compute-optimal training** — fix a FLOP budget, vary (N, D); H: optimum near ≈20 tokens/parameter (Chinchilla); needs ≫2 M-token data.

Also cheap ablations enabled by config switches: initialisation (depth-scaled vs flat 0.02; code is in `GPT.__init__`), GELU variant,
weight tying on/off (`model.tie_embeddings`), AdamW vs SGD (`training.optimizer`), dropout 0/0.1/0.2.

## Statistical hygiene

Differences smaller than the spread across seeds are noise; run ≥3 seeds for the headline comparisons, keep `best.pt` selection on
validation only, and touch the test set once per configuration at the end.
