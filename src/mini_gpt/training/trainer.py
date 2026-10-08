"""Training loop: gradient accumulation, mixed precision, clipping, periodic eval / checkpoint / samples / logging."""
from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from pathlib import Path

import torch

from mini_gpt.config import Config
from mini_gpt.evaluation.evaluation import format_samples, generate_fixed_samples
from mini_gpt.evaluation.perplexity import evaluate_loader
from mini_gpt.model.gpt import GPT
from mini_gpt.training.checkpoint import load_checkpoint, prune_step_checkpoints, save_checkpoint
from mini_gpt.training.metrics import plot_training
from mini_gpt.training.optimizer import build_optimizer
from mini_gpt.training.scheduler import WarmupCosineSchedule
from mini_gpt.utils.device import PrecisionInfo
from mini_gpt.utils.logging import MetricsLogger
from mini_gpt.utils.profiling import memory_stats, reset_peak, sync
from mini_gpt.utils.seed import set_rng_state

log = logging.getLogger("mini_gpt")


def make_autocast(prec: PrecisionInfo):
    if prec.mixed_precision and prec.amp_dtype is not None:
        return torch.autocast(device_type=prec.device.type, dtype=prec.amp_dtype)
    return contextlib.nullcontext()


def probe_max_micro_batch(model: GPT, cfg: Config, prec: PrecisionInfo, start: int) -> int:
    """Largest power-of-two-reduced micro-batch (<= start) whose forward+backward fits in memory.

    Only ``torch.cuda.OutOfMemoryError`` is caught; any other exception (a real bug) propagates.
    """
    device = prec.device
    bs = start
    T = cfg.model.context_length
    was = model.training
    model.train()
    while bs >= 1:
        try:
            model.zero_grad(set_to_none=True)
            x = torch.randint(0, cfg.model.vocab_size, (bs, T), device=device)
            with make_autocast(prec):
                _, loss, _ = model(x, x)
            loss.backward()
            sync(device)
            model.zero_grad(set_to_none=True)
            model.train(was)
            return bs
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            log.warning("OOM at micro-batch %d, trying %d", bs, bs // 2)
            bs //= 2
    raise RuntimeError("Even micro-batch size 1 does not fit in GPU memory; reduce context_length or model size.")


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: GPT,
        tokenizer,
        train_loader,
        val_loader,
        test_loader,
        run_dir: Path,
        prec: PrecisionInfo,
        resume: str | None = None,
        use_tensorboard: bool = True,
    ):
        self.cfg, self.model, self.tokenizer = cfg, model, tokenizer
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.run_dir, self.prec, self.device = Path(run_dir), prec, prec.device
        t = cfg.training
        self.ckpt_dir = Path(t.checkpoint_dir)
        self.model.to(self.device)
        self.model.set_gradient_checkpointing(t.gradient_checkpointing)
        self.optimizer = build_optimizer(model, t, self.device)
        self.scheduler = WarmupCosineSchedule(self.optimizer, t.learning_rate, t.min_learning_rate, t.warmup_steps, t.max_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=prec.use_grad_scaler)
        self.metrics = MetricsLogger(self.run_dir, tb_dir=Path(cfg.output_dir) / "logs" / self.run_dir.name, use_tensorboard=use_tensorboard)
        self.step, self.epoch_float, self.best_val = 0, 0.0, float("inf")
        self.micro_seen, self.loader_epochs = 0, 0
        self.start_time = time.perf_counter()
        self.prior_elapsed = 0.0
        self.tokens_seen = 0
        if resume:
            self._resume(resume)

    # ------------------------------------------------------------------ resume / save
    def _resume(self, path: str) -> None:
        ckpt = load_checkpoint(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step, self.epoch_float, self.best_val = ckpt["step"], ckpt["epoch"], ckpt["best_val_loss"]
        extra = ckpt.get("extra", {})
        self.tokens_seen = extra.get("tokens_seen", 0)
        self.prior_elapsed = extra.get("elapsed", 0.0)
        try:
            set_rng_state(ckpt["rng_state"])
        except Exception as exc:  # RNG layout can differ across machines (e.g. #GPUs); not fatal
            log.warning("Could not restore RNG state: %s", exc)
        log.info("Resumed from %s at step %d (best val loss %.4f)", path, self.step, self.best_val)
        log.info("Note: data order after resume is re-shuffled (not bit-identical to an uninterrupted run).")

    def _save(self, name: str) -> Path:
        path = self.ckpt_dir / name
        save_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.scaler, self.step, self.epoch_float,
            self.best_val, self.cfg, self.tokenizer.info() if self.tokenizer else None,
            extra={"tokens_seen": self.tokens_seen, "elapsed": self._elapsed()},
        )
        return path

    def _elapsed(self) -> float:
        return self.prior_elapsed + (time.perf_counter() - self.start_time)

    # ------------------------------------------------------------------ data
    def _batches(self):
        while True:
            for batch in self.train_loader:
                yield batch
            self.loader_epochs += 1

    # ------------------------------------------------------------------ eval / samples
    def evaluate(self, loader=None, full: bool = False) -> dict:
        """Periodic evaluation uses the first ``eval_iters`` batches (a fixed prefix => comparable across steps);
        ``full=True`` evaluates the entire split (used for the final validation/test numbers)."""
        loader = loader or self.val_loader
        n = self.cfg.training.eval_iters
        max_batches = None if (full or n <= 0) else n
        return evaluate_loader(self.model, loader, self.device, max_batches=max_batches, autocast_ctx=make_autocast(self.prec))

    def sample(self, tag: str | int) -> None:
        if self.tokenizer is None:
            return
        samples = generate_fixed_samples(self.model, self.tokenizer, self.device, self.cfg.training.sample_max_new_tokens,
                                         autocast_ctx=make_autocast(self.prec))
        text = format_samples(samples, tag)
        (self.run_dir / "samples" / f"step_{tag}.txt").write_text(text, encoding="utf-8")
        log.info("Saved samples for step %s -> samples/step_%s.txt", tag, tag)

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict:
        cfg, t = self.cfg, self.cfg.training
        T = cfg.model.context_length
        accum = t.gradient_accumulation_steps
        tokens_per_update = t.batch_size * accum * T
        samples_at = set(t.sample_steps)
        n_batches = max(1, len(self.train_loader))
        batches = self._batches()
        self.model.train()
        reset_peak(self.device)

        if self.step == 0:
            ev = self.evaluate()
            log.info("step 0 | val loss %.4f | val ppl %.1f (random-init baseline ~ ln(V) = %.3f)", ev["loss"], ev["perplexity"],
                     math.log(cfg.model.vocab_size))
            self.metrics.log(0, val_loss=ev["loss"], val_ppl=ev["perplexity"], elapsed=self._elapsed())
            if 0 in samples_at:
                self.sample(0)

        window_t0, window_tokens, window_examples = time.perf_counter(), 0, 0
        sync(self.device)
        while self.step < t.max_steps:
            lr = self.scheduler.step(self.step)
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=self.device)
            for _ in range(accum):
                x, y = next(batches)
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with make_autocast(self.prec):
                    _, loss, _ = self.model(x, y)
                # Divide by `accum` so the *sum* of micro-batch gradients equals the gradient of the mean loss
                # over the effective batch.
                self.scaler.scale(loss / accum).backward()
                loss_sum += loss.detach() / accum
                self.micro_seen += 1
            self.scaler.unscale_(self.optimizer)  # no-op when the scaler is disabled
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), t.gradient_clip).item()
            train_loss = loss_sum.item()
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"Non-finite training loss ({train_loss}) at step {self.step}. Lower the LR or check data.")
            if not math.isfinite(grad_norm) and not self.scaler.is_enabled():
                raise FloatingPointError(f"Non-finite gradient norm at step {self.step}.")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
            self.tokens_seen += tokens_per_update
            window_tokens += tokens_per_update
            window_examples += t.batch_size * accum
            self.epoch_float = self.micro_seen / n_batches

            if self.step % t.log_interval == 0 or self.step == t.max_steps:
                sync(self.device)
                dt = max(time.perf_counter() - window_t0, 1e-9)
                mem = memory_stats(self.device)
                row = dict(train_loss=train_loss, lr=lr, grad_norm=grad_norm, tokens_per_sec=window_tokens / dt,
                           examples_per_sec=window_examples / dt, epoch=self.epoch_float, elapsed=self._elapsed(), **mem)
                self.metrics.log(self.step, **row)
                s = self.step
                self.metrics.scalar("Loss/train", train_loss, s); self.metrics.scalar("LearningRate", lr, s)
                self.metrics.scalar("GradientNorm", grad_norm, s); self.metrics.scalar("TokensPerSecond", row["tokens_per_sec"], s)
                self.metrics.scalar("GPU/AllocatedMemory", mem["alloc_mb"], s); self.metrics.scalar("GPU/ReservedMemory", mem["reserved_mb"], s)
                log.info("step %d/%d | loss %.4f | lr %.2e | gnorm %.2f | %.0f tok/s | epoch %.2f | vram alloc/res/peak %.0f/%.0f/%.0f MB | rss %.0f MB | %.0fs",
                         s, t.max_steps, train_loss, lr, grad_norm, row["tokens_per_sec"], self.epoch_float,
                         mem["alloc_mb"], mem["reserved_mb"], mem["peak_mb"], mem["cpu_rss_mb"], row["elapsed"])
                window_t0, window_tokens, window_examples = time.perf_counter(), 0, 0

            if self.step % t.eval_interval == 0 or self.step == t.max_steps:
                ev = self.evaluate()
                self.metrics.log(self.step, val_loss=ev["loss"], val_ppl=ev["perplexity"], elapsed=self._elapsed())
                self.metrics.scalar("Loss/validation", ev["loss"], self.step)
                log.info("step %d | VALIDATION loss %.4f | ppl %.2f | bits/token %.3f", self.step, ev["loss"], ev["perplexity"], ev["bits_per_token"])
                if ev["loss"] < self.best_val:
                    self.best_val = ev["loss"]
                    self._save("best.pt")
                window_t0 = time.perf_counter()  # exclude eval time from throughput

            if self.step in samples_at and self.step != 0:
                self.sample(self.step)
                window_t0 = time.perf_counter()
            if self.step % t.checkpoint_interval == 0 and self.step != t.max_steps:
                self._save("latest.pt")
                self._save(f"step_{self.step}.pt")
                prune_step_checkpoints(self.ckpt_dir, t.keep_last_checkpoints)
                window_t0 = time.perf_counter()

        return self._finish()

    def _finish(self) -> dict:
        t = self.cfg.training
        self._save("latest.pt")
        final_path = self._save(f"step_{self.step}.pt")
        prune_step_checkpoints(self.ckpt_dir, t.keep_last_checkpoints)
        # Full validation + test passes with the final weights.
        val = self.evaluate(self.val_loader, full=True) if self.val_loader is not None else {}
        test = self.evaluate(self.test_loader, full=True) if self.test_loader is not None else {}
        self.sample("final")
        history = self.metrics.rows
        train_rows = [r for r in history if "train_loss" in r]
        mem = memory_stats(self.device)
        summary = {
            "steps": self.step,
            "tokens_seen": self.tokens_seen,
            "elapsed_sec": self._elapsed(),
            "final_train_loss": train_rows[-1]["train_loss"] if train_rows else None,
            "final_val": val,
            "final_test": test,
            "best_val_loss": self.best_val,
            "mean_tokens_per_sec": (sum(r["tokens_per_sec"] for r in train_rows) / len(train_rows)) if train_rows else None,
            "peak_vram_mb": mem["peak_mb"],
            "peak_cpu_rss_mb": max((r.get("cpu_rss_mb", 0) for r in train_rows), default=0.0),
            "final_checkpoint": str(final_path),
            "best_checkpoint": str(self.ckpt_dir / "best.pt"),
        }
        self.metrics.log(self.step, val_loss_full=val.get("loss"), test_loss_full=test.get("loss"))
        self.metrics.save(extra={"summary": summary})
        plot_dir = Path(self.cfg.output_dir) / "plots" / self.run_dir.name
        plot_training(history, plot_dir)
        plot_training(history, self.run_dir / "plots")
        self.metrics.close()
        log.info("Finished. Summary: %s", json.dumps({k: v for k, v in summary.items() if k not in ("final_val", "final_test")}, default=str))
        log.info("Final VALIDATION: %s | TEST: %s", val, test)
        return summary
