import torch
import torch.nn as nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight matrices (embedding, attention, FFN, output) get weight decay;
    RMSNorm weights (1-D) do not."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95), eps: float = 1e-8):
    # fused=True is only available for CUDA tensors; use the default implementation on CPU
    on_cuda = next(model.parameters()).is_cuda
    return torch.optim.AdamW(param_groups(model, weight_decay), lr=lr, betas=betas, eps=eps, fused=True if on_cuda else None)
