#!/usr/bin/env python
"""Train a mini GPT-2.  Example:  python scripts/train.py --config configs/small.yaml [--resume checkpoints/latest.pt]"""
import argparse
import json
import logging
import math
import platform
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
import yaml

from mini_gpt.config import apply_overrides_from_cli, load_config
from mini_gpt.data.dataloader import make_loader
from mini_gpt.data.preprocessing import prepare_dataset
from mini_gpt.data.splits import ensure_tokenized
from mini_gpt.model.gpt import GPT
from mini_gpt.tokenizer.tokenizer import BPETokenizer, tokenizer_dir_for
from mini_gpt.training.trainer import Trainer, probe_max_micro_batch
from mini_gpt.utils.device import detect_precision, get_device, hardware_summary
from mini_gpt.utils.logging import create_run_dir, git_commit, setup_logger
from mini_gpt.utils.parameter_count import estimate_flops, estimate_memory_mb, format_parameter_report, human
from mini_gpt.utils.seed import set_seed


def banner(cfg, hw, prec, model, tokens_per_update, mem_est) -> str:
    m, t = cfg.model, cfg.training
    gpu = hw.get("gpu", "none")
    lines = [
        "=" * 50, "Mini GPT-2", "=" * 50, "",
        "Device:",
        f"    CUDA available: {hw['cuda_available']}",
        f"    GPU: {gpu}",
        f"    VRAM: ~{hw.get('vram_gb', 'n/a')} GB",
        f"    CUDA version: {hw['cuda_version']}",
        f"    PyTorch version: {hw['pytorch']}",
        f"    CPU cores: {hw.get('cpu_count')}  |  System RAM: {hw.get('system_ram_gb', 'n/a')} GB", "",
        "Precision:",
        f"    BF16 supported: {prec.bf16_supported}",
        f"    FP16 supported: {prec.fp16_supported}",
        f"    Selected dtype: {prec.dtype_name}" + (f"  ({prec.note})" if prec.note else ""),
        f"    GradScaler: {prec.use_grad_scaler}", "",
        "Model:",
        f"    Parameters: {model.num_parameters():,}",
        f"    Context length: {m.context_length}",
        f"    Layers: {m.num_layers}",
        f"    Heads: {m.num_heads}",
        f"    Hidden dimension: {m.embedding_dim}",
        f"    Attention: {m.attention_impl}  |  Positions: {m.position_encoding}  |  Tied embeddings: {m.tie_embeddings}", "",
        "Training:",
        f"    Batch size (micro): {t.batch_size}",
        f"    Gradient accumulation: {t.gradient_accumulation_steps}",
        f"    Effective batch size: {t.batch_size * t.gradient_accumulation_steps} sequences",
        f"    Effective tokens/update: {tokens_per_update:,}",
        f"    Max steps: {t.max_steps}  ->  {tokens_per_update * t.max_steps:,} training tokens",
        f"    Est. training memory (rough): {mem_est['total']:.0f} MB "
        f"[params {mem_est['parameters']:.0f} + grads {mem_est['gradients']:.0f} + adam {mem_est['optimizer_states']:.0f} "
        f"+ acts {mem_est['activations']:.0f} + logits {mem_est['logits']:.0f}]",
        "=" * 50,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from, e.g. checkpoints/latest.pt")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true", help="request deterministic algorithms (slower)")
    ap.add_argument("--auto-batch-size", action="store_true", help="on CUDA OOM, halve the micro-batch (keeps effective batch)")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-tensorboard", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="section.key=value (YAML-parsed), e.g. training.learning_rate=6e-4")
    args = ap.parse_args()

    overrides = apply_overrides_from_cli(args.override)
    if args.seed is not None:
        overrides["training.seed"] = args.seed
    if args.max_steps is not None:
        overrides["training.max_steps"] = args.max_steps
    if args.checkpoint_dir:
        overrides["training.checkpoint_dir"] = args.checkpoint_dir
    if args.run_name:
        overrides["experiment_name"] = args.run_name
    cfg = load_config(args.config, overrides)
    warnings = cfg.validate()

    set_seed(cfg.training.seed, deterministic=args.deterministic)
    device = get_device(args.device)
    prec = detect_precision(device, cfg.precision.mixed_precision, cfg.precision.dtype)
    hw = hardware_summary(device)

    run_dir = create_run_dir(cfg.output_dir, cfg.experiment_name)
    logger = setup_logger("mini_gpt", run_dir / "training.log")
    for w in warnings:
        logger.warning("CONFIG WARNING: %s", w)

    # ---- data + tokenizer
    ds = cfg.data.dataset
    proc = Path(cfg.data.processed_dir) / ds
    if not (proc / "train.txt").exists():
        logger.info("Processed text not found; preparing dataset %s ...", ds)
        prepare_dataset(ds, cfg.data.raw_dir, cfg.data.processed_dir, cfg.data.custom_path)
    tok_dir = tokenizer_dir_for(cfg.data.tokenizer_dir, ds, cfg.tokenizer.vocab_size)
    tokenizer = BPETokenizer.load(tok_dir)  # raises a clear error if missing
    if tokenizer.vocab_size != cfg.model.vocab_size:
        logger.info("Setting model.vocab_size %d -> %d (tokenizer's actual size)", cfg.model.vocab_size, tokenizer.vocab_size)
        cfg.model.vocab_size = tokenizer.vocab_size
    bins = ensure_tokenized(ds, cfg.data.processed_dir, tokenizer)
    cfg.validate()

    model = GPT(cfg.model).to(device)
    T = cfg.model.context_length

    if args.auto_batch_size:
        if device.type == "cuda":
            eff = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
            bs = probe_max_micro_batch(model, cfg, prec, cfg.training.batch_size)
            if bs != cfg.training.batch_size:
                cfg.training.batch_size = bs
                cfg.training.gradient_accumulation_steps = math.ceil(eff / bs)
                logger.warning("auto-batch-size: micro-batch -> %d, grad accumulation -> %d (effective batch %d)",
                               bs, cfg.training.gradient_accumulation_steps, bs * cfg.training.gradient_accumulation_steps)
        else:
            logger.warning("--auto-batch-size only applies to CUDA; ignored on CPU")

    t = cfg.training
    tokens_per_update = t.batch_size * t.gradient_accumulation_steps * T
    mem_est = estimate_memory_mb(cfg.model, t.batch_size, efficient_attention=cfg.model.attention_impl != "manual",
                                 gradient_checkpointing=t.gradient_checkpointing, optimizer=t.optimizer)
    if device.type == "cuda" and mem_est["total"] > hw.get("vram_gb", 1e9) * 1024 * 0.9:
        logger.warning("Estimated training memory %.0f MB is close to/above available VRAM (%.1f GB). "
                       "Reduce batch_size, context_length, or enable gradient_checkpointing.", mem_est["total"], hw["vram_gb"])
    logger.info("\n%s", banner(cfg, hw, prec, model, tokens_per_update, mem_est))
    logger.info("\n%s", format_parameter_report(model))
    fl = estimate_flops(cfg.model, tokens_per_update * t.max_steps)
    logger.info("Approx. total training compute: %s FLOPs (model-based estimate, not measured)", human(fl["total_flops"]))

    pin = device.type == "cuda"
    train_loader = make_loader(bins["train"], T, t.batch_size, shuffle=True, max_tokens=cfg.data.max_train_tokens,
                               num_workers=cfg.data.num_workers, seed=t.seed, drop_last=True, pin_memory=pin)
    val_loader = make_loader(bins["validation"], T, t.batch_size, shuffle=False, max_tokens=cfg.data.max_eval_tokens, pin_memory=pin)
    test_loader = make_loader(bins["test"], T, t.batch_size, shuffle=False, max_tokens=cfg.data.max_eval_tokens, pin_memory=pin)
    n_train_tokens = len(train_loader.dataset.tokens)
    logger.info("Training tokens available: %s | per epoch: %d micro-batches | planned: %.2f epochs",
                f"{n_train_tokens:,}", len(train_loader), tokens_per_update * t.max_steps / max(n_train_tokens, 1))

    # ---- record the run
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    env = {"git_commit": git_commit(), "seed": t.seed, "hardware": hw, "precision": prec.dtype_name,
           "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
           "command": " ".join(sys.argv), "tokenizer": tokenizer.info(), "parameters": model.num_parameters(),
           "effective_tokens_per_update": tokens_per_update}
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))

    trainer = Trainer(cfg, model, tokenizer, train_loader, val_loader, test_loader, run_dir, prec,
                      resume=args.resume, use_tensorboard=not args.no_tensorboard)
    summary = trainer.train()
    logger.info("Run directory: %s", run_dir)
    print(json.dumps({k: v for k, v in summary.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
