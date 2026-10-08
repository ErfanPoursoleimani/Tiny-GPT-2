# Memory and compute on a 6 GB GPU

## What occupies VRAM during training (fp32 master weights, AdamW)

| item | size | default config (N = 23.0 M) |
|---|---|---|
| parameters | 4 N bytes | 88 MB |
| gradients | 4 N | 88 MB |
| AdamW `m` and `v` | 8 N | 176 MB |
| activations saved for backward | grows with `B·T·C·L` (+ `B·H·T²·L` if attention is materialised) | ~150–240 MB at B=8, T=256 |
| logits (+ softmax grad) in fp32 | `2·4·B·T·V` | ~500 MB at B=8 |
| CUDA context, allocator fragmentation, workspace | — | several hundred MB |

AdamW needs far more than the weights alone because it keeps **two** extra full-size tensors (first and second moment) and
backward needs a full gradient copy: 4 (w) + 4 (g) + 8 (m,v) = 16 bytes/parameter before activations — 4× the weight
memory. SGD without momentum would need 8.

`scripts/inspect_model.py` prints these estimates for any config; they are rough (Korthikanti et al. 2022 style
activation formula) and exclude allocator overhead. **Measure** with `scripts/benchmark.py` / the `vram alloc/res/peak`
fields in the training log. `allocated` = live tensors, `reserved` = what PyTorch's caching allocator holds from the driver,
`peak` = max allocated since reset.

## Levers (in order of what to try)

1. `training.batch_size` down + `gradient_accumulation_steps` up (same effective batch, same tokens/update).
2. `attention.implementation: sdpa/auto` (no T×T materialisation).
3. BF16/FP16 autocast (activations in 16-bit).
4. `training.gradient_checkpointing: true` (VRAM ↓, time ↑).
5. Smaller `context_length` (activations linear, attention quadratic in T).
6. Smaller vocab (logits `B·T·V`).

`python scripts/train.py --auto-batch-size` halves the micro-batch on CUDA OOM (and increases accumulation to keep the
effective batch). It catches only `torch.cuda.OutOfMemoryError`.

## FLOPs (approximation)

Training FLOPs per token ≈ `6·N_matmul + 12·L·C·T` (forward 2, backward 4 FLOPs per weight per token, plus attention
score/value matmuls). Default: ~145 MFLOPs/token → 10,000 steps × 16,384 tokens ≈ 2.4×10¹⁶ FLOPs. This is a model-based
estimate, not a hardware measurement; the common `6·N·D` rule gives 2.3×10¹⁶. A laptop RTX 3050 sustains only a fraction
of its peak in practice, so measure tokens/sec rather than trusting peak TFLOPs.
