import math

import pytest
import torch
import torch.nn.functional as F

from model import (
    CausalSelfAttention,
    FeedForward,
    RoPE,
    TransformerBlock,
    TransformerLM,
    generate_ids,
    naive_causal_attention,
    top_p_filter,
)

CFG = dict(vocab_size=10_000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, rope_theta=10_000.0)


def small_model(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(vocab_size=100, context_length=32, d_model=64, num_layers=2, num_heads=4, d_ff=128)
    cfg.update(kw)
    return TransformerLM(**cfg)


# ------------------------------------------------------------------ attention
@pytest.mark.parametrize("shape", [(5, 7, 16), (2, 3, 7, 16)])  # 3-D and 4-D inputs
def test_naive_attention_matches_sdpa(shape):
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape) for _ in range(3))
    ours = naive_causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert ours.shape == ref.shape
    assert torch.allclose(ours, ref, atol=1e-5)


def test_naive_attention_is_causal():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 6, 8) for _ in range(3))
    out1 = naive_causal_attention(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, 4:] += 10
    v2[:, 4:] -= 10
    out2 = naive_causal_attention(q, k2, v2)
    assert torch.equal(out1[:, :4], out2[:, :4])
    assert not torch.allclose(out1[:, 4:], out2[:, 4:])
    # the first position can only attend to itself
    assert torch.allclose(out1[:, 0], v[:, 0])


# ------------------------------------------------------------------ RoPE
def slow_rope(x, theta):
    """Builds the full d x d block-diagonal rotation matrix for every position."""
    seq, d = x.shape[-2], x.shape[-1]
    out = torch.empty_like(x)
    for i in range(seq):
        R = torch.zeros(d, d, dtype=torch.float64)
        for k in range(d // 2):
            a = i / theta ** (2 * k / d)
            R[2 * k, 2 * k], R[2 * k, 2 * k + 1] = math.cos(a), -math.sin(a)
            R[2 * k + 1, 2 * k], R[2 * k + 1, 2 * k + 1] = math.sin(a), math.cos(a)
        out[..., i, :] = (x[..., i, :].double() @ R.T).to(x.dtype)
    return out


def test_rope_matches_rotation_matrix():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 11, 8)
    rope = RoPE(10_000.0, 8, 64)
    assert torch.allclose(rope(x), slow_rope(x, 10_000.0), atol=1e-5)


def test_rope_properties():
    torch.manual_seed(0)
    rope = RoPE(10_000.0, 16, 64)
    x = torch.randn(4, 10, 16)
    y = rope(x)
    assert torch.allclose(y[:, 0], x[:, 0])  # position 0 is not rotated
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)  # rotations keep length
    # relative position: <R_i q, R_j k> depends only on i - j
    q, k = torch.randn(16), torch.randn(16)
    seq = torch.stack([q] * 40)
    kseq = torch.stack([k] * 40)
    rq, rk = rope(seq), rope(kseq)
    d1 = rq[10] @ rk[7]
    d2 = rq[35] @ rk[32]
    d3 = rq[10] @ rk[2]
    assert torch.allclose(d1, d2, atol=1e-4)
    assert not torch.allclose(d1, d3, atol=1e-3)
    assert "cos" not in rope.state_dict()  # non-persistent buffer


# ------------------------------------------------------------------ shapes
@pytest.mark.parametrize("lead", [(7,), (2, 7), (2, 3, 7)])
def test_pieces_keep_shape(lead):
    torch.manual_seed(0)
    x = torch.randn(*lead, 64)
    assert CausalSelfAttention(64, 4, max_len=32)(x).shape == x.shape
    assert FeedForward(64, 128)(x).shape == x.shape
    assert TransformerBlock(64, 4, 128, max_len=32)(x).shape == x.shape
    xh = torch.randn(*lead, 16)
    assert RoPE(10_000.0, 16, 32)(xh).shape == xh.shape


def test_swiglu_formula():
    torch.manual_seed(0)
    ff = FeedForward(8, 12)
    x = torch.randn(3, 8)
    expected = (F.silu(x @ ff.w1.weight.T) * (x @ ff.w3.weight.T)) @ ff.w2.weight.T
    assert torch.allclose(ff(x), expected, atol=1e-6)


def test_logits_shape_full_config():
    model = TransformerLM(**CFG)
    ids = torch.randint(0, 10_000, (2, 17))
    assert model(ids).shape == (2, 17, 10_000)


# ------------------------------------------------------------------ causality
def test_model_cannot_see_the_future():
    model = small_model().eval()
    torch.manual_seed(1)
    ids = torch.randint(0, 100, (2, 20))
    t = 9
    changed = ids.clone()
    changed[:, t + 1 :] = torch.randint(0, 100, (2, 20 - t - 1))
    with torch.no_grad():
        a, b = model(ids), model(changed)
    assert torch.equal(a[:, : t + 1], b[:, : t + 1])  # exactly the same up to position t
    assert not torch.allclose(a[:, t + 1 :], b[:, t + 1 :])  # and the future did change


# ------------------------------------------------------------------ init loss
def test_initial_loss_close_to_ln_vocab():
    # At initialisation the logits are small and nearly the same for every token, so the
    # predicted distribution is close to uniform over 10,000 tokens. The cross-entropy of a
    # uniform distribution is -log(1/V) = ln(V) = ln(10000) ~= 9.21, whatever the target is.
    torch.manual_seed(0)
    model = TransformerLM(**CFG)
    ids = torch.randint(0, 10_000, (4, 64))
    with torch.no_grad():
        logits = model(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 10_000), ids[:, 1:].reshape(-1))
    assert abs(loss.item() - math.log(10_000)) < 0.3


# ------------------------------------------------------------------ parameter count
def test_parameter_count_matches_hand_count():
    V, d, L, dff = 10_000, 512, 4, 1344
    embedding = V * d  # 5,120,000
    per_layer = 4 * d * d + 3 * d * dff + 2 * d  # attn + SwiGLU + 2 RMSNorm weights = 3,113,984
    final_norm = d
    lm_head = d * V  # 5,120,000
    hand = embedding + L * per_layer + final_norm + lm_head
    assert hand == 22_696_448
    model = TransformerLM(**CFG)
    assert sum(p.numel() for p in model.parameters()) == hand
    assert sum(p.numel() for p in model.blocks.parameters()) == L * per_layer == 12_455_936


# ------------------------------------------------------------------ generation
def test_top_p_keeps_smallest_set():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    out = top_p_filter(probs, 0.7)  # 0.5 < 0.7, 0.5+0.3 >= 0.7 -> keep first two
    assert torch.allclose(out, torch.tensor([0.5, 0.3, 0.0, 0.0]) / 0.8)
    out = top_p_filter(probs, 0.1)  # the top token is always kept
    assert torch.allclose(out, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    shuffled = torch.tensor([0.15, 0.5, 0.05, 0.3])
    assert torch.allclose(top_p_filter(shuffled, 0.7), torch.tensor([0.0, 0.5, 0.0, 0.3]) / 0.8)


def test_greedy_generation_is_deterministic():
    model = small_model()
    a = generate_ids(model, [1, 2, 3], 20, temperature=0)
    b = generate_ids(model, [1, 2, 3], 20, temperature=0)
    assert a == b and len(a) == 20


def test_sampling_reproducible_with_seed_and_stops_at_eos():
    model = small_model()
    g1, g2 = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    a = generate_ids(model, [1], 30, temperature=1.0, top_p=0.9, generator=g1)
    b = generate_ids(model, [1], 30, temperature=1.0, top_p=0.9, generator=g2)
    assert a == b
    # with eos = the greedy first token, greedy decoding must stop immediately
    first = generate_ids(model, [1], 1, temperature=0)[0]
    assert generate_ids(model, [1], 30, temperature=0, eos_id=first) == []


def test_generation_handles_prompt_longer_than_context():
    model = small_model()  # context_length 32
    out = generate_ids(model, list(range(1, 50)), 5, temperature=0)
    assert len(out) == 5
