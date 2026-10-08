#!/usr/bin/env python
"""Print parameter counts, memory estimates and approximate FLOPs for a config (no data needed)."""
import argparse

import _bootstrap  # noqa: F401

from mini_gpt.config import apply_overrides_from_cli, load_config
from mini_gpt.model.gpt import GPT
from mini_gpt.utils.parameter_count import estimate_flops, estimate_memory_mb, format_parameter_report, human


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, apply_overrides_from_cli(args.override))
    for w in cfg.validate():
        print("WARNING:", w)
    model = GPT(cfg.model)
    print(format_parameter_report(model))
    t = cfg.training
    for impl_label, eff in (("manual attention (materialises T x T)", False), ("fused SDPA", True)):
        m = estimate_memory_mb(cfg.model, t.batch_size, efficient_attention=eff, gradient_checkpointing=t.gradient_checkpointing, optimizer=t.optimizer)
        print(f"\nEstimated training memory, micro-batch {t.batch_size}, {impl_label}  [rough estimate]")
        for k, v in m.items():
            print(f"  {k:<16}{v:>10.1f} MB")
    tokens = t.batch_size * t.gradient_accumulation_steps * cfg.model.context_length
    fl = estimate_flops(cfg.model, tokens * t.max_steps)
    print(f"\nApprox. FLOPs/token: {human(fl['flops_per_token'])}  | total for {t.max_steps} steps "
          f"({tokens:,} tokens/update): {human(fl['total_flops'])}  (6*N*D rule: {human(fl['six_n_d_rule'])})  [approximation]")
    print(f"Effective batch: {t.batch_size} x {t.gradient_accumulation_steps} = {t.batch_size * t.gradient_accumulation_steps} sequences "
          f"= {tokens:,} tokens/update")


if __name__ == "__main__":
    main()
