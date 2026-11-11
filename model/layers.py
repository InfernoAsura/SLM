"""Building blocks: hand-written reference attention, RoPE, causal self-attention, SwiGLU."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def naive_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d_k) + causal mask) V, written out by hand.

    q, k, v: (..., seq, d_k). Every "future" position (key index > query index) is set
    to -inf before the softmax so it gets exactly zero weight.
    Used only as a reference in the tests; the model uses F.scaled_dot_product_attention.
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    L, S = q.shape[-2], k.shape[-2]
    future = torch.ones(L, S, dtype=torch.bool, device=q.device).triu(diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class RoPE(nn.Module):
    """Rotary position embedding.

    The pair (x_{2k}, x_{2k+1}) at position i is rotated by angle i / theta^(2k / d_head).
    cos/sin tables are precomputed once and stored as non-persistent buffers (they are not
    saved in the state_dict, they are rebuilt from the config).
    """

    def __init__(self, theta: float, d_head: int, max_len: int):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even"
        inv_freq = theta ** (-torch.arange(0, d_head, 2, dtype=torch.float64) / d_head)  # (d_head/2,)
        angles = torch.arange(max_len, dtype=torch.float64)[:, None] * inv_freq[None, :]  # (max_len, d_head/2)
        self.register_buffer("cos", angles.cos().float(), persistent=False)
        self.register_buffer("sin", angles.sin().float(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """x: (..., seq, d_head). positions: optional (seq,) integer positions (default 0..seq-1)."""
        seq = x.shape[-2]
        if positions is None:
            cos, sin = self.cos[:seq], self.sin[:seq]
        else:
            cos, sin = self.cos[positions], self.sin[positions]
        x_even, x_odd = x[..., 0::2].float(), x[..., 1::2].float()
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # interleave back: (..., seq, d/2, 2) -> (..., seq, d)
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int = 2048, rope_theta: float = 10_000.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(rope_theta, self.d_head, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., seq, d_model) -> (..., seq, d_model)."""
        h = self.num_heads
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=h)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=h)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=h)
        q, k = self.rope(q), self.rope(k)  # RoPE on queries and keys only
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = rearrange(out, "... h s d -> ... s (h d)")
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU: W2 (SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
