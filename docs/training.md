# Training

## Objective: next-token prediction

For a stream `x₁…x_N`, the model defines `P(x_t | x_<t)`. The training objective is the average negative
log-likelihood (cross-entropy):

```
L = −(1/N) Σ_t log P(x_t | x_<t)
```

Implementation: for a chunk starting at `s`, `x = tokens[s : s+T]` and `y = tokens[s+1 : s+T+1]`. Position `t` of the
input is *the last token the model is allowed to see* when it predicts `y[t] = tokens[s+t+1]`; the target is one ahead
because the label of "what comes after token t" is token t+1. The model's logits at position *t* only depend on
`x[0..t]` (causal mask), so there is no leakage even though all positions are computed in one pass.
```python
loss = F.cross_entropy(logits.view(-1, V), y.view(-1))     # mean over B·T tokens, computed in ≥fp32
```
**Training objective** (what we optimise: cross-entropy on train data) ≠ **evaluation metric** (loss/perplexity on
held-out data) ≠ **generation strategy** (greedy / temperature / top-k / top-p sampling at inference). A lower loss
does not by itself say anything about which decoding strategy gives nicer text.

## Backpropagation

`loss.backward()` applies the chain rule through the computation graph in reverse, producing `∂L/∂θ` for every
parameter (and storing activations needed along the way — the main training memory cost besides optimizer states).
Softmax+cross-entropy has the clean gradient `∂L/∂logits = softmax(logits) − onehot(y)`. We verify autograd against
float64 central finite differences (`python scripts/benchmark.py --gradient-check`; also in `pytest`).

## AdamW

Per parameter `θ` with gradient `g`: `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`,
`θ ← θ − lr·( m̂/(√v̂+ε) + λθ )`. Weight decay `λ` is **decoupled** (applied to θ directly rather than added to the
gradient where Adam's adaptive scaling would distort it). `β₂=0.95` (vs 0.999) reacts faster to changing gradient
scale, a common choice for LM training. Decay is applied only to ≥2-D tensors; biases and LayerNorm parameters are
excluded (decaying LN gains would pull them toward 0, away from their identity initialisation).
Memory: `m` and `v` are two fp32 tensors per parameter, so AdamW needs ~3× the weight memory *beyond* the weights, plus
a gradient copy: 16 bytes/parameter total in fp32 (see memory.md).

## Learning-rate schedule

Linear warmup `0→peak` over `warmup_steps` (lr = peak·(step+1)/warmup), then cosine decay to `min_learning_rate`:
`lr = min + ½(peak−min)(1+cos(π·progress))`. Warmup avoids destructive early updates while Adam's second-moment
estimates are still noisy; cosine decay gives a smooth anneal. The schedule is a pure function of the step, so resuming
is exact.

## Gradient accumulation

Per optimizer step we run `accum` micro-batches, each contributing `loss/accum`; gradients add up in `.grad` so the
sum equals the gradient of the mean loss over `batch_size × accum` sequences (tested). Report:
`effective_batch = batch_size × accum` sequences, `tokens/update = effective_batch × context_length`
(default 64 × 256 = 16,384). Clipping (`gradient_clip`, default 1.0) uses the global norm *after* accumulation and
unscaling; the logged `grad_norm` is the pre-clip norm.

## Mixed precision

`detect_precision` queries the device: BF16 if `torch.cuda.is_bf16_supported()`, else FP16 (with `GradScaler`, which
multiplies the loss to avoid gradient underflow and skips steps with inf/NaN), else FP32. Under `torch.autocast` matmuls
run in half precision while reductions (softmax, LayerNorm) and our loss stay in fp32. Weights and optimizer states stay
fp32 ("master weights"). BF16 has FP32's exponent range, so it needs no scaler; FP16 does. A non-finite loss raises
instead of silently continuing.

## Gradient checkpointing (optional)

`training.gradient_checkpointing: true` wraps each block in `torch.utils.checkpoint`: only block inputs are stored and
activations are recomputed during backward — less VRAM at the price of an extra forward pass per block (typically tens of percent more time; measure it with the benchmark). Gradients are identical (tested).

## Run outputs

`outputs/runs/<timestamp>_<exp>/`: `config.yaml`, `environment.json` (git commit, seed, hardware, versions, command),
`training.log`, `metrics.json` (history + summary), `samples/step_*.txt`, `plots/`; TensorBoard in `outputs/logs/<run>`;
checkpoints in `training.checkpoint_dir` (`latest.pt`, `best.pt`, `step_N.pt`, last 3 kept).

## Token budget

Tokens seen = `steps × tokens/update`. Default: 10,000 × 16,384 ≈ 164 M tokens, i.e. dozens of epochs over WikiText-2.
For a 23 M-parameter model the compute-optimal rule of thumb (~20 tokens/parameter, Hoffmann et al. 2022) is ~460 M
tokens *of diverse data*; repeating a 2 M-token corpus 60+ times will overfit. This project prioritises learning and
controlled experiments over capability.
