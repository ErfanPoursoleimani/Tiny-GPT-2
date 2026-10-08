# Architecture

Notation: `B` batch, `T` sequence length, `C` embedding dim, `H` heads, `D = C/H` head dim, `V` vocab, `L` layers.

## Data flow and shapes

| Stage | Tensor | Shape |
|---|---|---|
| token ids | `idx` | `(B,T)` int64 |
| embeddings | `x = Drop(E[idx] + P[0..T-1])` | `(B,T,C)` |
| each block | `x ← x + Attn(LN₁(x))`, `x ← x + MLP(LN₂(x))` | `(B,T,C)` |
| final norm | `LN_f(x)` | `(B,T,C)` |
| logits | `x W_E^T` (tied) | `(B,T,V)` |

*Parameters* are learned tensors (`E, P, W_Q…`). *Activations* are the intermediate tensors above (recomputed every
batch, stored for backward). *Gradients* are `∂loss/∂parameter`. *Logits* are unnormalised scores; *probabilities*
are `softmax(logits)`; the *loss* is a scalar.

## LayerNorm

For each token vector `x ∈ R^C`: `LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β`, with `μ, σ²` computed over the C features
of that token only (not across the batch). Pre-LN places LN *inside* the residual branch, leaving a clean identity
path `x → x + …` from input to output, which makes deep stacks trainable without careful warmup. (The original
Transformer used post-LN; GPT-2 moved to pre-LN and added a final LN.)

## MLP

`MLP(x) = W₂ · GELU(W₁ x + b₁) + b₂` with `W₁: C→4C`, `W₂: 4C→C`, applied independently to every position. GELU is
`x·Φ(x)`; `activation: gelu` uses the exact erf form, `gelu_new` the tanh approximation GPT-2 used. We never
substitute ReLU. Roughly two thirds of block parameters are in the MLP (8C² of 12C²).

## Residual connections

Each sub-layer *adds* its output to the stream. Gradients flow through the `+` unchanged, so early layers receive
a direct gradient signal; sub-layers learn *corrections* to the stream.

## Initialisation (and the decision)

All Linear/Embedding weights `N(0, 0.02²)`, biases 0, LN `(1,0)`. **Decision:** residual-output projections
(`attn.c_proj`, `mlp.c_proj`) use `std = 0.02/√(2L)`. Reason: each block adds two terms to the residual stream; if
independent with equal variance σ², the stream variance after L blocks is ≈ (1 + 2L)σ². Shrinking by √(2L) keeps it
near-constant at init. This is what the GPT-2 paper describes (it scales by 1/√N with N the number of residual
layers). `tests/test_model.py::test_init_statistics` checks it. Whether this beats a flat 0.02 for a 6-layer model
is a small effect and is listed as an experiment (docs/experiments.md).

## Weight tying

`lm_head.weight` *is* `tok_emb.weight`, so the output scores are dot products between the final hidden state and each
token's embedding. Saves `V·C` parameters (12.3 M of 35.3 M by default). Gradients from both uses accumulate in the
same tensor. Configurable: `model.tie_embeddings`.

## Positional information

`learned`: `P ∈ R^{T_max×C}` added to token embeddings (hard cap at `context_length`). `rope`: no additive term;
queries and keys are rotated in each attention layer (see attention.md). Both share the same model class; the
choice is a config switch.

## Parameter accounting (default config)

Token embedding 12,288,000 + position embedding 98,304 + 6 blocks × 1,774,464 + final LN 768 = **23,033,856**,
where a block = attention (4C²+4C = 591,360) + MLP (8C²+5C = 1,181,568) + two LayerNorms (4C = 1,536).
`scripts/inspect_model.py` prints the categorised table; `tests/test_model.py` checks the count against a closed-form
formula for 16 config variants (tied/untied × learned/RoPE × bias/no-bias × …).
