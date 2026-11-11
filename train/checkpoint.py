import random

import numpy as np
import torch


def unwrap(model):
    """torch.compile wraps the model; save/load the original module so keys have no prefix."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(path, model, optimizer, scheduler, scaler, step: int, rng: np.random.Generator | None = None, extra: dict | None = None):
    state = {
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "batch_rng": rng.bit_generator.state if rng is not None else None,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, rng: np.random.Generator | None = None, map_location="cpu") -> dict:
    """Restores everything in place and returns the checkpoint dict (with 'step' and 'extra')."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    unwrap(model).load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    # RNG states must be CPU ByteTensors even when map_location moved everything to the GPU
    torch.set_rng_state(state["torch_rng"].cpu())
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    random.setstate(state["python_rng"])
    if rng is not None and state.get("batch_rng") is not None:
        rng.bit_generator.state = state["batch_rng"]
    return state
