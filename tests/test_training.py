import json
import math

import numpy as np
import pytest
import torch

from mini_gpt.config import Config, ModelConfig, config_from_dict, load_config
from mini_gpt.model.gpt import GPT
from mini_gpt.training.checkpoint import load_checkpoint, model_from_checkpoint, save_checkpoint
from mini_gpt.training.optimizer import build_optimizer
from mini_gpt.training.scheduler import WarmupCosineSchedule
from mini_gpt.utils.device import detect_precision
from mini_gpt.utils.parameter_count import estimate_flops, estimate_memory_mb


def test_scheduler_shape():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    s = WarmupCosineSchedule(opt, peak_lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    lrs = [s.lr_at(i) for i in range(0, 120)]
    assert lrs[0] > 0 and all(a < b for a, b in zip(lrs[:9], lrs[1:10]))  # strictly increasing warmup
    assert abs(lrs[9] - 1.0) < 1e-9 and abs(lrs[10] - 1.0) < 1e-9          # reaches peak at end of warmup
    assert all(a >= b for a, b in zip(lrs[10:110], lrs[11:111]))           # monotone cosine decay
    assert abs(lrs[110] - 0.1) < 1e-9 and abs(lrs[119] - 0.1) < 1e-9       # floor
    s.step(50); assert opt.param_groups[0]["lr"] == pytest.approx(s.lr_at(50))


def test_gradient_accumulation_equals_big_batch(tiny_cfg):
    torch.manual_seed(1)
    m = GPT(tiny_cfg).train()
    x = torch.randint(0, tiny_cfg.vocab_size, (4, 16)); y = torch.randint(0, tiny_cfg.vocab_size, (4, 16))
    m.zero_grad(); m(x, y)[1].backward()
    big = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    for i in range(2):  # 2 micro-batches of 2, each loss divided by accumulation steps
        (m(x[2 * i : 2 * i + 2], y[2 * i : 2 * i + 2])[1] / 2).backward()
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(big, m.parameters()))


def test_optimizer_groups(tiny_cfg):
    from mini_gpt.config import TrainingConfig
    m = GPT(tiny_cfg)
    opt = build_optimizer(m, TrainingConfig(weight_decay=0.1), torch.device("cpu"))
    assert isinstance(opt, torch.optim.AdamW)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
    assert opt.param_groups[0]["betas"] == (0.9, 0.95)


def test_checkpoint_roundtrip_identical_outputs(tiny_cfg, tmp_path):
    cfg = Config(model=tiny_cfg)
    m = GPT(tiny_cfg)
    opt = build_optimizer(m, cfg.training, torch.device("cpu"))
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    m(x, x)[1].backward(); opt.step()
    sched = WarmupCosineSchedule(opt, 1e-3, 1e-4, 5, 50); sched.step(7)
    save_checkpoint(tmp_path / "c.pt", m, opt, sched, None, 7, 0.5, 1.23, cfg, {"vocab_size": 100})
    m2, cfg2, ck = model_from_checkpoint(tmp_path / "c.pt", torch.device("cpu"))
    m.eval(); m2.eval()
    with torch.no_grad():
        assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-7)
    assert ck["step"] == 7 and ck["best_val_loss"] == 1.23 and ck["seed"] == 42 and ck["tokenizer"] == {"vocab_size": 100}
    assert cfg2.model == tiny_cfg and ck["scheduler"]["last_step"] == 7 and "rng_state" in ck
    assert not list(tmp_path.glob("*.tmp"))


def test_config_yaml_inheritance_and_validation(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  embedding_dim: 64\n  num_heads: 4\nattention:\n  implementation: manual\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nmodel:\n  num_layers: 3\nposition_encoding:\n  type: rope\n")
    cfg = load_config(tmp_path / "child.yaml", {"training.learning_rate": 5e-4})
    assert (cfg.model.embedding_dim, cfg.model.num_layers, cfg.model.attention_impl, cfg.model.position_encoding) == (64, 3, "manual", "rope")
    assert cfg.training.learning_rate == 5e-4
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"modle": {}})
    bad = Config(model=ModelConfig(embedding_dim=10, num_heads=3))
    with pytest.raises(ValueError, match="divisible"):
        bad.validate()


def test_repo_configs_load():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "configs"
    for f in list(root.glob("*.yaml")) + list(root.glob("experiments/*.yaml")):
        load_config(f).validate()
    assert load_config(root / "tiny.yaml").model.embedding_dim == 192
    assert load_config(root / "small.yaml").training.effective_batch_size == 64


def test_precision_detection_cpu_fallback():
    p = detect_precision(torch.device("cpu"), True, "bfloat16")
    assert p.amp_dtype is None and not p.use_grad_scaler and not p.mixed_precision


def test_estimators_sane():
    cfg = ModelConfig()
    a = estimate_memory_mb(cfg, 8, efficient_attention=False)
    b = estimate_memory_mb(cfg, 8, efficient_attention=True)
    c = estimate_memory_mb(cfg, 8, gradient_checkpointing=True)
    assert a["activations"] > b["activations"] > c["activations"]
    assert a["optimizer_states"] == pytest.approx(2 * a["parameters"])  # Adam: 2 fp32 states per parameter
    fl = estimate_flops(cfg, 1_000_000)
    assert 0.8 < fl["total_flops"] / fl["six_n_d_rule"] < 1.3


def test_overfit_tiny_dataset():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("overfit", Path(__file__).resolve().parents[1] / "scripts" / "overfit.py")
    import sys
    sys.path.insert(0, str(Path(spec.origin).parent))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    res = mod.run_overfit(steps=150, verbose=False)
    assert res["final_loss"] < 0.1 < res["initial_loss"]
    assert res["checkpoint_max_logit_diff"] < 1e-6
    assert res["generation"].startswith("the cat sat on the mat")


def test_end_to_end_smoke_training_and_resume(tmp_path):
    from mini_gpt.data.dataloader import make_loader
    from mini_gpt.data.splits import ensure_tokenized
    from mini_gpt.tokenizer.train_tokenizer import train_bpe_tokenizer
    from mini_gpt.training.trainer import Trainer
    from mini_gpt.utils.logging import create_run_dir, setup_logger

    text = "the cat sat on the mat. a dog barked at the moon. " * 400
    d = tmp_path / "proc" / "toy"; d.mkdir(parents=True)
    for s in ("train", "validation", "test"):
        (d / f"{s}.txt").write_text(text)
    tok = train_bpe_tokenizer(texts=[text], vocab_size=300, min_frequency=1)
    bins = ensure_tokenized("toy", tmp_path / "proc", tok)

    cfg = config_from_dict({
        "model": {"vocab_size": tok.vocab_size, "context_length": 32, "embedding_dim": 32, "num_layers": 2, "num_heads": 2, "dropout": 0.0},
        "training": {"batch_size": 4, "gradient_accumulation_steps": 2, "max_steps": 30, "warmup_steps": 3, "learning_rate": 3e-3,
                     "min_learning_rate": 3e-4, "eval_interval": 10, "eval_iters": 3, "log_interval": 5, "checkpoint_interval": 10,
                     "sample_steps": [0, 20], "sample_max_new_tokens": 5, "checkpoint_dir": str(tmp_path / "ckpt")},
        "output_dir": str(tmp_path / "out"), "experiment_name": "smoke",
    })
    T = 32
    mk = lambda split, sh: make_loader(bins[split], T, 4, shuffle=sh, drop_last=sh)
    run_dir = create_run_dir(cfg.output_dir, "smoke"); setup_logger("mini_gpt", run_dir / "training.log")
    prec = detect_precision(torch.device("cpu"))
    tr = Trainer(cfg, GPT(cfg.model), tok, mk("train", True), mk("validation", False), mk("test", False), run_dir, prec)
    summary = tr.train()

    hist = json.loads((run_dir / "metrics.json").read_text())["history"]
    train_losses = [r["train_loss"] for r in hist if "train_loss" in r]
    assert train_losses[-1] < train_losses[0] - 0.5            # loss decreases
    n_val = len(mk("validation", False).dataset) * 32
    assert summary["final_val"]["tokens"] == n_val and summary["final_val"]["tokens"] > 3 * 4 * 32   # full split, not eval_iters batches
    assert math.isfinite(summary["final_val"]["perplexity"]) and summary["final_test"]["loss"] > 0
    assert (tmp_path / "ckpt" / "latest.pt").exists() and (tmp_path / "ckpt" / "best.pt").exists()
    assert (run_dir / "samples" / "step_0.txt").exists() and (run_dir / "samples" / "step_final.txt").exists()
    assert (tmp_path / "out" / "plots" / run_dir.name / "training_loss.png").exists()
    assert any((tmp_path / "out" / "logs" / run_dir.name).iterdir())  # TensorBoard events

    # resume: step/optimizer/weights restored, and training can continue
    ck = load_checkpoint(tmp_path / "ckpt" / "latest.pt")
    assert ck["step"] == 30
    cfg.training.max_steps = 36
    run2 = create_run_dir(cfg.output_dir, "smoke_resume")
    tr2 = Trainer(cfg, GPT(cfg.model), tok, mk("train", True), mk("validation", False), mk("test", False), run2, prec,
                  resume=str(tmp_path / "ckpt" / "latest.pt"), use_tensorboard=False)
    assert tr2.step == 30
    for a, b in zip(tr2.model.parameters(), ck["model"].values()):
        pass
    assert torch.allclose(tr2.model.state_dict()["transformer.ln_f.weight"], ck["model"]["transformer.ln_f.weight"])
    s2 = tr2.train()
    assert s2["steps"] == 36


def test_gradient_check_float64():
    from mini_gpt.evaluation.benchmarks import gradient_check
    cfg = ModelConfig(vocab_size=40, context_length=8, embedding_dim=16, num_layers=2, num_heads=2, dropout=0.0, attention_impl="manual")
    res = gradient_check(cfg, n_checks=10)
    assert all(r["passed"] for r in res), res
    cfg.position_encoding = "rope"; cfg.attention_impl = "sdpa"
    assert all(r["passed"] for r in gradient_check(cfg, n_checks=10, seed=1))


def test_cli_override_parsing():
    from mini_gpt.config import apply_overrides_from_cli
    o = apply_overrides_from_cli(["training.learning_rate=6e-4", "model.context_length=512", "attention.implementation=sdpa",
                                  "training.sample_steps=[0,10]", "model.tie_embeddings=false"])
    assert o["training.learning_rate"] == 6e-4 and isinstance(o["training.learning_rate"], float)
    assert o["model.context_length"] == 512 and o["attention.implementation"] == "sdpa"
    assert o["training.sample_steps"] == [0, 10] and o["model.tie_embeddings"] is False
