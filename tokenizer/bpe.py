"""Byte-level BPE training.

train_bpe(input_path, vocab_size, special_tokens) -> (vocab, merges)

The work is split into two clearly separated phases so it can be profiled:
  1. pretokenize_file(): split on special tokens, run the GPT-2 regex, count pre-tokens
     (optionally in parallel over chunks of the file that end on a special token).
  2. merge_pairs(): repeatedly merge the most frequent neighbouring pair, updating
     only the pre-tokens that actually contained that pair.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)


# ----------------------------------------------------------------------------- phase 1
def split_on_special(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on the special tokens (escaped!) and drop the special tokens themselves."""
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    return re.split(pattern, text)


def count_pretokens(text: str, special_tokens: list[str]) -> Counter:
    """Count GPT-2 pre-tokens (as str) in text. Special tokens are never counted."""
    counts: Counter = Counter()
    for segment in split_on_special(text, special_tokens):
        counts.update(m.group() for m in PAT_RE.finditer(segment))
    return counts


def _next_split(f, pos: int, size: int, split_token: bytes) -> int:
    """Offset of the first split_token at or after pos (or size if there is none)."""
    f.seek(pos)
    while True:
        block = f.read(1 << 16)
        if not block:
            return size
        idx = block.find(split_token)
        if idx != -1:
            return pos + idx
        if len(block) <= len(split_token):
            return size
        # keep an overlap so a token split across two blocks is still found
        pos += len(block) - len(split_token)
        f.seek(pos)


def find_chunk_boundaries(
    path: str, num_chunks: int, split_token: bytes, max_bytes: int | None = None
) -> list[int]:
    """Byte offsets that cut the file (or its first ~max_bytes) into ~num_chunks pieces.
    Every cut is placed at the start of a split_token, so no story is ever cut in half."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        end = size if max_bytes is None or max_bytes >= size else _next_split(f, max_bytes, size, split_token)
        boundaries = [0]
        for i in range(1, num_chunks):
            pos = _next_split(f, end * i // num_chunks, end, split_token)
            if boundaries[-1] < pos < end:
                boundaries.append(pos)
    boundaries.append(end)
    return boundaries


def _count_chunk(args) -> Counter:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return count_pretokens(text, special_tokens)


def pretokenize_file(
    path: str, special_tokens: list[str], num_workers: int = 1, max_bytes: int | None = None
) -> Counter:
    """Pre-token counts for a whole file. Uses multiprocessing when num_workers > 1.
    max_bytes limits training to the first max_bytes of the file (cut at a special token)."""
    split_token = (special_tokens[0] if special_tokens else "\n").encode("utf-8")
    num_chunks = max(1, num_workers * 8)
    bounds = find_chunk_boundaries(path, num_chunks, split_token, max_bytes)
    jobs = [(path, s, e, special_tokens) for s, e in zip(bounds[:-1], bounds[1:])]
    total: Counter = Counter()
    if num_workers <= 1:
        for job in jobs:
            total.update(_count_chunk(job))
    else:
        with Pool(num_workers) as pool:
            for c in pool.imap_unordered(_count_chunk, jobs):
                total.update(c)
    return total


# ----------------------------------------------------------------------------- phase 2
def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes], new: bytes) -> tuple[bytes, ...]:
    out = []
    i, n = 0, len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def merge_pairs(
    word_counts: dict[tuple[bytes, ...], int], num_merges: int
) -> list[tuple[bytes, bytes]]:
    """Run num_merges BPE merges over pre-tokens given as tuples of byte tokens.

    Ties on frequency are broken by the lexicographically larger pair (as Python tuples).
    After each merge only the words that contained the merged pair are updated.
    """
    words = list(word_counts.keys())
    freqs = [word_counts[w] for w in words]
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            pair_to_words[p].add(wi)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        new_token = best[0] + best[1]
        merges.append(best)
        for wi in list(pair_to_words.get(best, ())):
            w, f = words[wi], freqs[wi]
            for p in zip(w, w[1:]):  # remove all old pair counts of this word
                c = pair_counts.get(p)
                if c is not None:
                    c -= f
                    if c <= 0:
                        del pair_counts[p]
                    else:
                        pair_counts[p] = c
                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(wi)
            new_w = _merge_word(w, best, new_token)
            words[wi] = new_w
            for p in zip(new_w, new_w[1:]):  # add the new pair counts back
                pair_counts[p] += f
                pair_to_words[p].add(wi)
        pair_counts.pop(best, None)
        pair_to_words.pop(best, None)
    return merges


# ----------------------------------------------------------------------------- public API
def train_bpe_from_counts(
    pretoken_counts: Counter, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError("vocab_size is smaller than 256 + number of special tokens")
    word_counts: dict[tuple[bytes, ...], int] = {}
    for tok, c in pretoken_counts.items():
        key = tuple(bytes([b]) for b in tok.encode("utf-8"))
        word_counts[key] = word_counts.get(key, 0) + c
    merges = merge_pairs(word_counts, num_merges)
    for a, b in merges:
        vocab[len(vocab)] = a + b
    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 1,
    max_bytes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer. Returns (vocab: id -> bytes, merges: list of pairs)."""
    counts = pretokenize_file(input_path, special_tokens, num_workers=num_workers, max_bytes=max_bytes)
    return train_bpe_from_counts(counts, vocab_size, special_tokens)
