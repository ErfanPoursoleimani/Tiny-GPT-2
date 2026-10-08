"""Optimizer construction with decay / no-decay parameter groups."""
from __future__ import annotations

import torch

from mini_gpt.config import TrainingConfig
from mini_gpt.model.gpt import GPT


def build_optimizer(model: GPT, tcfg: TrainingConfig, device: torch.device) -> torch.optim.Optimizer:
    """AdamW(betas=(0.9, 0.95), eps=1e-8). Only >=2-D tensors are weight-decayed (see GPT.configure_param_groups).

    AdamW *decouples* weight decay from the gradient-based update: w <- w - lr * (adam_step + wd * w),
    unlike L2-regularised Adam where the penalty gradient would be rescaled by the adaptive denominator.
    Memory: AdamW stores two extra fp32 tensors per parameter (first and second moments), i.e. 8 bytes/param
    on top of 4 bytes of weights and 4 of gradients -> ~16 bytes/param before activations.

    SGD (momentum 0.9) is provided for educational comparison.
    """
    groups = model.configure_param_groups(tcfg.weight_decay)
    if tcfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=tcfg.learning_rate, momentum=0.9)
    use_fused = device.type == "cuda"
    return torch.optim.AdamW(groups, lr=tcfg.learning_rate, betas=(tcfg.beta1, tcfg.beta2), eps=tcfg.eps, fused=use_fused or None)
