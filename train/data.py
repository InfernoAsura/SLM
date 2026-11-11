import numpy as np
import torch


def load_tokens(path: str) -> np.ndarray:
    """Memory-map a uint16 token file written with np.save (only touched pages are read)."""
    return np.load(path, mmap_mode="r")


def batch_from_starts(data: np.ndarray, starts, context_length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = np.stack([np.asarray(data[i : i + context_length + 1]) for i in starts]).astype(np.int64)
    x = torch.from_numpy(chunk[:, :-1])
    y = torch.from_numpy(chunk[:, 1:])  # targets are the inputs shifted one step right
    device = torch.device(device)
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_batch(data: np.ndarray, batch_size: int, context_length: int, device, rng: np.random.Generator | None = None):
    """Random starting points i in [0, len - context_length - 1]; inputs data[i:i+T],
    targets data[i+1:i+T+1]. The largest start keeps the last target inside the array."""
    rng = rng if rng is not None else np.random.default_rng()
    starts = rng.integers(0, len(data) - context_length, size=batch_size)
    return batch_from_starts(data, starts, context_length, device)


def fixed_eval_starts(n_tokens: int, num_batches: int, batch_size: int, context_length: int, seed: int = 1234) -> np.ndarray:
    """The same validation batches every time (same seed -> same start positions)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_tokens - context_length, size=(num_batches, batch_size))
