"""Autoregressive text generation with temperature and top-p (nucleus) sampling."""
from __future__ import annotations

import torch


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of most-likely tokens whose total probability is >= top_p,
    zero out the rest and renormalise. probs: (..., vocab)."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # a token is kept if the probability mass *before* it is still < top_p
    # (so the token that crosses top_p is included, and the top token is always kept)
    keep_sorted = (cumulative - sorted_probs) < top_p
    keep = torch.zeros_like(keep_sorted).scatter(-1, sorted_idx, keep_sorted)
    filtered = probs * keep
    return filtered / filtered.sum(dim=-1, keepdim=True)


@torch.no_grad()
def generate_ids(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    min_new_tokens: int = 0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Returns only the newly generated ids (the eos token itself is not included).

    temperature <= 0 means greedy decoding. min_new_tokens > 0 forbids eos until that many
    tokens exist (used only to get long sample stories; it is off by default).
    """
    model.eval()
    device = next(model.parameters()).device
    context_length = model.config.context_length
    ids = list(prompt_ids)
    new: list[int] = []
    for _ in range(max_new_tokens):
        window = ids[-context_length:]  # only feed the last context_length tokens
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        if eos_id is not None and len(new) < min_new_tokens:
            logits[eos_id] = float("-inf")
        if temperature <= 0:
            next_id = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs = top_p_filter(probs, top_p)
            next_id = int(torch.multinomial(probs, 1, generator=generator))
        if eos_id is not None and next_id == eos_id:
            break
        ids.append(next_id)
        new.append(next_id)
    return new


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    min_new_tokens: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[str, int]:
    """Generate a continuation of prompt. Returns (prompt + continuation text, number of new tokens)."""
    eos_id = tokenizer.special_ids.get("<|endoftext|>")
    prompt_ids = tokenizer.encode(prompt)
    new = generate_ids(model, prompt_ids, max_new_tokens, temperature, top_p, eos_id, min_new_tokens, generator)
    return tokenizer.decode(prompt_ids + new), len(new)
