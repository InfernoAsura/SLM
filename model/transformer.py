"""Pre-norm Transformer block and the full decoder-only language model."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .layers import CausalSelfAttention, FeedForward


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10_000.0

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerBlock(nn.Module):
    """z = x + Attn(RMSNorm(x));  y = z + FFN(RMSNorm(z))."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_len: int = 2048, rope_theta: float = 10_000.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(d_model, num_heads, max_len=max_len, rope_theta=rope_theta)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x + self.attn(self.norm1(x))
        return z + self.ffn(self.norm2(z))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        self.config = ModelConfig(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, max_len=context_length, rope_theta=rope_theta) for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @classmethod
    def from_config(cls, cfg: ModelConfig | dict) -> "TransformerLM":
        if isinstance(cfg, dict):
            keys = ModelConfig.__dataclass_fields__.keys()
            cfg = ModelConfig(**{k: cfg[k] for k in keys if k in cfg})
        return cls(**asdict(cfg))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (batch, seq) int64 -> logits (batch, seq, vocab_size)."""
        assert ids.shape[-1] <= self.config.context_length, "sequence longer than context_length"
        x = self.token_emb(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
