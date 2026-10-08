import math

import pytest
import torch

from mini_gpt.config import ModelConfig
from mini_gpt.model.gpt import GPT
from mini_gpt.utils.parameter_count import analytic_param_count, count_parameters


def test_forward_shapes_and_finite_loss(tiny_cfg):
    m = GPT(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (4, 16)); y = torch.randint(0, tiny_cfg.vocab_size, (4, 16))
    logits, loss, _ = m(x, y)
    assert logits.shape == (4, 16, tiny_cfg.vocab_size)
    assert torch.isfinite(loss)


def test_initial_loss_near_log_vocab(tiny_cfg):
    m = GPT(tiny_cfg).eval()
    x = torch.randint(0, tiny_cfg.vocab_size, (16, 16)); y = torch.randint(0, tiny_cfg.vocab_size, (16, 16))
    with torch.no_grad():
        loss = m(x, y)[1].item()
    assert abs(loss - math.log(tiny_cfg.vocab_size)) < 0.3  # small init => near-uniform predictions


def test_backward_all_params_get_gradients(tiny_cfg):
    m = GPT(tiny_cfg)
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    m(x, x)[1].backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing
    assert all(p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "pos_emb" not in n)


def test_weight_tying():
    cfg = ModelConfig(vocab_size=50, context_length=8, embedding_dim=16, num_layers=1, num_heads=2, tie_embeddings=True)
    m = GPT(cfg)
    assert m.lm_head.weight is m.embeddings.tok_emb.weight
    untied = GPT(ModelConfig(**{**cfg.__dict__, "tie_embeddings": False}))
    assert untied.lm_head.weight is not untied.embeddings.tok_emb.weight
    assert untied.num_parameters() == m.num_parameters() + 50 * 16


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("pos", ["learned", "rope"])
@pytest.mark.parametrize("bias", [True, False])
def test_param_count_matches_formula(tie, pos, bias):
    cfg = ModelConfig(vocab_size=64, context_length=8, embedding_dim=16, num_layers=3, num_heads=2,
                      tie_embeddings=tie, position_encoding=pos, bias=bias)
    m = GPT(cfg)
    assert count_parameters(m)["total"] == analytic_param_count(cfg) == m.num_parameters()


def test_default_config_param_count():
    assert analytic_param_count(ModelConfig()) == 23_033_856


def test_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(embedding_dim=100, num_heads=6).validate()
    with pytest.raises(ValueError):
        GPT(ModelConfig(embedding_dim=100, num_heads=6))
    with pytest.raises(ValueError):
        ModelConfig(attention_impl="flash").validate()


def test_init_statistics(tiny_cfg):
    m = GPT(tiny_cfg)
    assert abs(m.embeddings.tok_emb.weight.std().item() - 0.02) < 0.004
    proj = m.transformer.blocks[0].attn.c_proj.weight.std().item()
    assert abs(proj - 0.02 / math.sqrt(2 * tiny_cfg.num_layers)) < 0.004   # depth-scaled residual projections
    ln = m.transformer.ln_f
    assert torch.all(ln.weight == 1) and torch.all(ln.bias == 0)


def test_param_groups_no_decay_for_1d(tiny_cfg):
    m = GPT(tiny_cfg)
    decay, no_decay = m.configure_param_groups(0.1)
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(m.parameters()))


def test_gelu_variants_differ_slightly(tiny_cfg):
    import copy
    c2 = copy.deepcopy(tiny_cfg); c2.activation = "gelu_new"
    m1, m2 = GPT(tiny_cfg).eval(), GPT(c2).eval(); m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    d = (m1(x)[0] - m2(x)[0]).abs().max().item()
    assert 0 < d < 1e-2


def test_gradient_checkpointing_same_grads(tiny_cfg):
    m = GPT(tiny_cfg).train()
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    m(x, x)[1].backward(); g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(); m.set_gradient_checkpointing(True)
    m(x, x)[1].backward(); g2 = [p.grad.clone() for p in m.parameters()]
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g1, g2))
