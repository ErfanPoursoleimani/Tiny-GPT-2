"""Linear warmup followed by cosine decay to a minimum learning rate."""
from __future__ import annotations

import math

import torch


class WarmupCosineSchedule:
    """lr(step) = peak * (step+1)/warmup                       for step < warmup
                 min + 0.5 (peak-min)(1+cos(pi * progress))    for warmup <= step < max_steps
                 min                                           afterwards
    (``step+1`` so the very first update uses a small *non-zero* LR.)
    The schedule is a pure function of the step, so its state is just the step counter: resuming is exact.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, peak_lr: float, min_lr: float, warmup_steps: int, max_steps: int):
        self.optimizer, self.peak, self.min = optimizer, peak_lr, min_lr
        self.warmup, self.max_steps = warmup_steps, max_steps
        self.last_step = -1

    def lr_at(self, step: int) -> float:
        if self.warmup > 0 and step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        if step >= self.max_steps:
            return self.min
        denom = max(1, self.max_steps - self.warmup)
        progress = (step - self.warmup) / denom
        return self.min + 0.5 * (self.peak - self.min) * (1.0 + math.cos(math.pi * progress))

    def step(self, step: int) -> float:
        lr = self.lr_at(step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.last_step = step
        return lr

    def state_dict(self) -> dict:
        return {"last_step": self.last_step, "peak": self.peak, "min": self.min, "warmup": self.warmup, "max_steps": self.max_steps}

    def load_state_dict(self, state: dict) -> None:
        self.last_step = state["last_step"]
