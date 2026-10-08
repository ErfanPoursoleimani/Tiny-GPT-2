# Causal multi-head self-attention

## Math

Given `X ∈ R^{B×T×C}`:

```
Q = X W_Q,  K = X W_K,  V = X W_V          each (B,T,C)
split heads: (B,T,C) → (B,T,H,D) → (B,H,T,D)
S = Q Kᵀ / √D                              (B,H,T,T)   scores
S_ij ← −∞ for j > i                        causal mask M
A = softmax_j(S)                           (B,H,T,T)   rows sum to 1; A_ij = 0 for j > i
Y_h = A V                                  (B,H,T,D)
Y = concat_h(Y_h) W_O                      (B,T,C)
```

`√D` keeps score variance ≈ 1 when q, k entries have unit variance; without it softmax saturates for large D and
gradients vanish. We fuse `W_Q, W_K, W_V` into one `Linear(C,3C)` (`c_attn`) and split the output.

### Causal mask

`M_ij = 0 if j ≤ i else −∞`. After softmax, `A_ij = 0` for `j > i`, so output *i* depends only on tokens ≤ *i*.
This is what permits training on all `T` positions in parallel while each position still predicts its next token
from the past only. **Test**: `tests/test_attention.py::test_future_tokens_do_not_affect_past` changes the last token
and asserts every earlier position's logits are unchanged (manual + SDPA, learned + RoPE); deliberately removing
the mask makes it fail.

## PyTorch mapping

```python
q, k, v = self.c_attn(x).split(C, dim=2)                       # (B,T,C) each
q = q.view(B,T,H,D).transpose(1,2)                             # (B,H,T,D)
att = (q @ k.transpose(-2,-1)) / math.sqrt(D)                  # (B,H,T,S)
att = att.masked_fill(~mask[S-T:S,:S], -inf).softmax(dim=-1)
y = (att @ v).transpose(1,2).contiguous().view(B,T,C)
```

(`S` = number of keys = `T` normally, `cache_len + T` with a KV cache.)

## Two implementations (`attention.implementation`)

| value | what runs | memory | notes |
|---|---|---|---|
| `manual` | the code above | materialises `B·H·T·T` scores + probs | transparent; used by tests; can store `last_attention` |
| `sdpa` | `F.scaled_dot_product_attention` | PyTorch may choose a fused (flash / mem-efficient) CUDA kernel or the math fallback; never forced to hold T×T | same results to ~1e-5 (tested) |
| `auto` | `sdpa` if available, else `manual` | | default |

We do not hand-write FlashAttention: PyTorch's SDPA already dispatches to one on supported GPUs/dtypes.
Which backend is used depends on hardware, dtype, mask and dropout; `benchmark.py` measures the actual speed and
peak memory difference on your machine.

## KV cache

During generation, keys/values of already-processed tokens never change (causal mask + deterministic positions), so
we store per-layer `(K,V)` of shape `(B,H,S,D)` and, for each new token, compute only its `q,k,v`, append `k,v`, and
attend over the cache: O(T) work per token instead of O(T²) for re-running the whole prefix. With RoPE the keys are
rotated *before* caching so cached entries stay valid. Learned absolute positions cannot go past `context_length`;
when the window is full `generate()` drops the cache and re-prefills the last `context_length-1` tokens (sliding window).
`tests/test_attention.py::test_kv_cache_matches_full_forward` verifies prefill + chunk + single-token decoding equals
the full forward pass; `benchmark.py` reports the speedup.

## RoPE (optional)

Each pair `(x_i, x_{i+D/2})` of q and k is rotated by angle `m·θ_i` (position `m`, `θ_i = 10000^{−2i/D}`). Then
`⟨R_m q, R_n k⟩` depends only on `m−n` (tested), giving relative position information without a learned table and
with graceful handling of cache offsets.
