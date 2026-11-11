"""One optimisation step with optional fp16 mixed precision, and the validation loop."""
import contextlib

import torch

from .loss import lm_loss


def autocast_ctx(device_type: str, dtype: str):
    """fp16 autocast on CUDA (works on T4 / Colab / Kaggle); plain fp32 otherwise.
    bf16 is deliberately not offered: the T4 has no fast bf16 support."""
    if dtype == "fp16" and device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(device_type: str, dtype: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=(dtype == "fp16" and device_type == "cuda"))


def train_step(model, x, y, optimizer, scheduler, scaler, dtype: str = "fp16", grad_clip: float = 1.0) -> dict:
    model.train()
    with autocast_ctx(x.device.type, dtype):
        logits = model(x)
    loss = lm_loss(logits, y)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    # Unscale BEFORE clipping, otherwise we would clip the (huge) scaled gradients.
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scale_before = scaler.get_scale() if scaler.is_enabled() else 1.0
    scaler.step(optimizer)  # skipped automatically if the gradients contain inf/NaN
    scaler.update()
    skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
    scheduler.step()
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "skipped": bool(skipped),
        "scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
    }


@torch.no_grad()
def evaluate(model, batches, dtype: str = "fp16") -> float:
    """Mean loss over a fixed list of (x, y) batches (or a callable that yields them)."""
    model.eval()
    total, n = 0.0, 0
    for x, y in batches() if callable(batches) else batches:
        with autocast_ctx(x.device.type, dtype):
            logits = model(x)
        total += lm_loss(logits, y).item()
        n += 1
    model.train()
    return total / max(n, 1)
