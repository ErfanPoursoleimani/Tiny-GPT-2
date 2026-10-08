import copy

import pytest
import torch

from mini_gpt.config import ModelConfig
from mini_gpt.model.attention import CausalSelfAttention
from mini_gpt.model.gpt import GPT


def make_attn(cfg, impl, pos="learned"):
    c = copy.deepcopy(cfg); c.attention_impl = impl; c.position_encoding = pos
    return CausalSelfAttention(c).eval()


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_shapes(tiny_cfg, impl):
    B, T, C, H = 3, 10, tiny_cfg.embedding_dim, tiny_cfg.num_heads
    a = make_attn(tiny_cfg, impl)
    a.store_attention = True
    y, _ = a(torch.randn(B, T, C))
    assert y.shape == (B, T, C)
    if impl == "manual":
        assert a.last_attention.shape == (B, H, T, T)


def test_attention_matrix_is_causal_and_normalised(tiny_cfg):
    a = make_attn(tiny_cfg, "manual"); a.store_attention = True
    a(torch.randn(2, 12, tiny_cfg.embedding_dim))
    att = a.last_attention
    assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)
    upper = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
    assert (att[..., upper] == 0).all()


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_future_tokens_do_not_affect_past(tiny_cfg, impl, pos):
    """Changing token t must leave outputs at positions < t unchanged (the core causality property)."""
    c = copy.deepcopy(tiny_cfg); c.attention_impl = impl; c.position_encoding = pos
    model = GPT(c).eval()
    a = torch.randint(0, c.vocab_size, (2, 8))
    b = a.clone()
    b[:, -1] = (b[:, -1] + 7) % c.vocab_size  # [a,b,c,...,X]
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)
    assert not torch.allclose(la[:, -1], lb[:, -1], atol=1e-5)  # the changed position itself does differ


def test_change_in_middle_only_affects_suffix(tiny_cfg):
    model = GPT(tiny_cfg).eval()
    a = torch.randint(0, tiny_cfg.vocab_size, (1, 10))
    b = a.clone(); b[0, 5] = (b[0, 5] + 1) % tiny_cfg.vocab_size
    with torch.no_grad():
        la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :5], lb[:, :5], atol=1e-5)
    assert not torch.allclose(la[:, 5:], lb[:, 5:], atol=1e-5)


@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_manual_matches_sdpa(tiny_cfg, pos):
    c1 = copy.deepcopy(tiny_cfg); c1.attention_impl = "manual"; c1.position_encoding = pos
    c2 = copy.deepcopy(c1); c2.attention_impl = "sdpa"
    m1, m2 = GPT(c1).eval(), GPT(c2).eval()
    m2.load_state_dict(m1.state_dict())
    x = torch.randint(0, c1.vocab_size, (3, 16))
    with torch.no_grad():
        assert torch.allclose(m1(x)[0], m2(x)[0], atol=1e-5)


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
@pytest.mark.parametrize("pos", ["learned", "rope"])
def test_kv_cache_matches_full_forward(tiny_cfg, impl, pos):
    c = copy.deepcopy(tiny_cfg); c.attention_impl = impl; c.position_encoding = pos
    model = GPT(c).eval()
    x = torch.randint(0, c.vocab_size, (2, 12))
    with torch.no_grad():
        full = model(x)[0]
        # prefill 7 tokens, then a 3-token chunk, then single tokens
        lg, _, past = model(x[:, :7], use_cache=True)
        parts = [lg]
        lg, _, past = model(x[:, 7:10], past_kvs=past, use_cache=True); parts.append(lg)
        for t in range(10, 12):
            lg, _, past = model(x[:, t : t + 1], past_kvs=past, use_cache=True); parts.append(lg)
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4)


def test_rope_relative_property():
    from mini_gpt.model.embeddings import RotaryEmbedding

    rope = RotaryEmbedding(8, 64)
    q = torch.randn(1, 1, 1, 8); k = torch.randn(1, 1, 1, 8)
    def dot(m, n):
        qm, _ = rope(q, k, offset=m)
        _, kn = rope(q, k, offset=n)
        return (qm * kn).sum().item()
    assert abs(dot(5, 2) - dot(15, 12)) < 1e-4  # depends only on m - n
    assert abs(dot(5, 2) - dot(5, 3)) > 1e-6


def test_sequence_longer_than_context_raises(tiny_cfg):
    model = GPT(tiny_cfg)
    with pytest.raises(ValueError):
        model(torch.randint(0, tiny_cfg.vocab_size, (1, tiny_cfg.context_length + 1)))
