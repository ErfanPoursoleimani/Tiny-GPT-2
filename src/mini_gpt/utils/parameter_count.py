"""Parameter counting, memory estimation and FLOPs approximation.

All estimates here are *approximations* for planning. Measure real VRAM with
``scripts/benchmark.py`` on your own GPU.
"""
from __future__ import annotations

import math

import torch.nn as nn

from mini_gpt.config import ModelConfig


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Categorised parameter counts. Tied weights are counted once (``parameters()`` dedups)."""
    cats = {"embedding": 0, "attention": 0, "mlp": 0, "layernorm": 0, "lm_head": 0, "other": 0}
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        if "embeddings" in name:
            cats["embedding"] += n
        elif ".attn." in name:
            cats["attention"] += n
        elif ".mlp." in name:
            cats["mlp"] += n
        elif "ln_" in name:
            cats["layernorm"] += n
        elif name.startswith("lm_head"):
            cats["lm_head"] += n
        else:
            cats["other"] += n
    cats["total"] = sum(cats.values())
    cats["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return cats


def format_parameter_report(model: nn.Module) -> str:
    c = count_parameters(model)
    lines = ["Model parameters", "----------------"]
    for k in ("embedding", "attention", "mlp", "layernorm", "lm_head", "other"):
        lines.append(f"{k.capitalize():<10}: {c[k]:>14,}")
    lines.append("----------------")
    lines.append(f"Total     : {c['total']:>14,}  (~{c['total'] / 1e6:.2f}M)")
    lines.append(f"Trainable : {c['trainable']:>14,}")
    if c["lm_head"] == 0:
        lines.append("(lm_head is tied to the token embedding, so it adds 0 extra parameters)")
    return "\n".join(lines)


def analytic_param_count(cfg: ModelConfig) -> int:
    """Closed-form parameter count (used to cross-check ``count_parameters`` in tests)."""
    C, L, V = cfg.embedding_dim, cfg.num_layers, cfg.vocab_size
    b = 1 if cfg.bias else 0
    tok = V * C
    pos = cfg.context_length * C if cfg.position_encoding == "learned" else 0
    attn = C * 3 * C + b * 3 * C + C * C + b * C
    mlp = C * 4 * C + b * 4 * C + 4 * C * C + b * C
    ln = 2 * 2 * C  # two LayerNorms (weight+bias) per block
    ln_f = 2 * C
    head = 0 if cfg.tie_embeddings else V * C
    return tok + pos + L * (attn + mlp + ln) + ln_f + head


def estimate_memory_mb(
    cfg: ModelConfig,
    batch_size: int,
    param_bytes: int = 4,
    activation_bytes: int = 2,
    efficient_attention: bool = True,
    gradient_checkpointing: bool = False,
    optimizer: str = "adamw",
) -> dict[str, float]:
    """Rough training-memory estimate in MB.

    * parameters   : N * param_bytes (fp32 master weights)
    * gradients    : N * param_bytes
    * optimizer    : AdamW keeps two fp32 moment tensors (m, v) => 2 * N * 4 bytes
                     (SGD without momentum: 0)
    * activations  : per-layer activations saved for backward. Following Korthikanti et al.
                     (2022), roughly ``B*T*C*34`` bytes (16-bit) per layer for the linear/LN/MLP
                     parts, plus ``5*B*H*T^2`` for the attention scores/probabilities/dropout mask
                     when attention is materialised (manual implementation). Fused SDPA kernels
                     avoid the T^2 term. Gradient checkpointing keeps ~one layer input per layer.
    * logits       : B*T*V in fp32 (logits + softmax grad) -- often the largest single tensor
                     for a 32k vocabulary!
    """
    N = analytic_param_count(cfg)
    B, T, C, H, L, V = batch_size, cfg.context_length, cfg.embedding_dim, cfg.num_heads, cfg.num_layers, cfg.vocab_size
    mb = 1024**2
    per_layer = B * T * C * 34 * (activation_bytes / 2)
    if not efficient_attention:
        per_layer += 5 * B * H * T * T * (activation_bytes / 2)
    acts = L * per_layer
    if gradient_checkpointing:
        acts = L * B * T * C * activation_bytes + per_layer  # layer inputs + one live layer
    logits = 2 * B * T * V * 4
    opt = 2 * N * 4 if optimizer == "adamw" else 0
    out = {
        "parameters": N * param_bytes / mb,
        "gradients": N * param_bytes / mb,
        "optimizer_states": opt / mb,
        "activations": acts / mb,
        "logits": logits / mb,
    }
    out["total"] = sum(out.values())
    return out


def estimate_flops(cfg: ModelConfig, tokens: int) -> dict[str, float]:
    """Approximate training FLOPs (forward+backward ~ 3x forward = 6 FLOPs/param/token).

    flops/token ~= 6 * N_matmul + 12 * L * C * T   (attention score & value matmuls)
    where N_matmul counts weights that participate in matmuls (blocks + LM head; lookups are free).
    This is a *model-based approximation*, not a hardware measurement.
    """
    C, L, V, T = cfg.embedding_dim, cfg.num_layers, cfg.vocab_size, cfg.context_length
    n_blocks = L * (4 * C * C + 8 * C * C)
    n_matmul = n_blocks + V * C
    per_token = 6 * n_matmul + 12 * L * C * T
    return {
        "parameters": float(analytic_param_count(cfg)),
        "sequence_length": float(T),
        "tokens": float(tokens),
        "flops_per_token": float(per_token),
        "total_flops": float(per_token) * tokens,
        "six_n_d_rule": 6.0 * analytic_param_count(cfg) * tokens,
    }


def human(n: float) -> str:
    if n <= 0:
        return "0"
    units = ["", "K", "M", "G", "T", "P", "E"]
    i = min(int(math.log10(n) // 3), len(units) - 1)
    return f"{n / 1000 ** i:.2f}{units[i]}"
