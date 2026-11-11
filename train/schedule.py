import math

from torch.optim.lr_scheduler import LambdaLR


def lr_at(t: int, max_lr: float, min_lr: float, warmup_steps: int, cosine_steps: int) -> float:
    """Linear warm-up to max_lr over warmup_steps (T_w), cosine decay to min_lr at step
    cosine_steps (T_c), then constant min_lr."""
    if t < warmup_steps:
        return t / warmup_steps * max_lr
    if t <= cosine_steps:
        progress = (t - warmup_steps) / max(1, cosine_steps - warmup_steps)
        return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    return min_lr


def build_scheduler(optimizer, max_lr: float, min_lr: float, warmup_steps: int, cosine_steps: int) -> LambdaLR:
    """LambdaLR multiplies the base lr of the optimizer (= max_lr) by the returned factor."""
    return LambdaLR(optimizer, lambda t: lr_at(t, max_lr, min_lr, warmup_steps, cosine_steps) / max_lr)
