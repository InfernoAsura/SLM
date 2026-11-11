import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy. logits (B, T, V), targets (B, T).
    Computed in fp32 even when the logits come out of an fp16 autocast region."""
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V).float(), targets.reshape(-1))
