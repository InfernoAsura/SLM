"""Byte-level BPE Tokenizer: encode / encode_iterable / decode, plus saving and loading.

File formats (both unambiguous, because tokens may contain spaces or invalid UTF-8):
  vocab.json  {"<id>": "<hex of the token bytes>", ...}
  merges.txt  one merge per line: "<hex of left> <hex of right>"
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

import regex as re

from .bpe import PAT_RE


def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str) -> None:
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(i): tok.hex() for i, tok in sorted(vocab.items())}, f, indent=0)
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")


def load_tokenizer_files(vocab_path: str, merges_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {int(i): bytes.fromhex(h) for i, h in json.load(f).items()}
    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                a, b = line.split(" ")
                merges.append((bytes.fromhex(a), bytes.fromhex(b)))
    return vocab, merges


class Tokenizer:
    _CACHE_LIMIT = 1_000_000

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.special_tokens: list[str] = list(special_tokens or [])
        self.token_to_id: dict[bytes, int] = {tok: i for i, tok in self.vocab.items()}
        # add special tokens that are not in the vocab yet to the end
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in self.token_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = b
                self.token_to_id[b] = new_id
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: r for r, pair in enumerate(merges)}
        self.special_ids: dict[str, int] = {st: self.token_to_id[st.encode("utf-8")] for st in self.special_tokens}
        if self.special_tokens:
            # longest first, so that when special tokens overlap the longer one wins
            alts = "|".join(re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True))
            self._special_re = re.compile(f"({alts})")
            self._max_special_len = max(len(t) for t in self.special_tokens)
        else:
            self._special_re = None
            self._max_special_len = 0
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        vocab, merges = load_tokenizer_files(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ encoding
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached
        parts = [bytes([b]) for b in pretoken.encode("utf-8")]
        ranks = self.merge_ranks
        # Applying the earliest-learned merge first, again and again, is the same as
        # replaying all merges in the order they were learned.
        while len(parts) > 1:
            best_rank, best_pair = None, None
            for pair in zip(parts, parts[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            a, b = best_pair
            merged, i, n = [], 0, len(parts)
            while i < n:
                if i < n - 1 and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = [self.token_to_id[p] for p in parts]
        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.clear()
        self._cache[pretoken] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for m in PAT_RE.finditer(text):
            ids.extend(self._encode_pretoken(m.group()))
        return ids

    def encode(self, text: str) -> list[int]:
        if self._special_re is None:
            return self._encode_ordinary(text)
        ids: list[int] = []
        # re.split with a capturing group keeps the special tokens at the odd positions
        for i, piece in enumerate(self._special_re.split(text)):
            if i % 2 == 1:
                ids.append(self.special_ids[piece])
            elif piece:
                ids.extend(self._encode_ordinary(piece))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode a stream of strings (e.g. an open file). Memory stays bounded:
        we only hold back text whose pre-tokenisation could still change when more text
        arrives (the last pre-token, and anything that could be the start of a special token)."""
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            # 1) everything up to and including the last complete special token is final
            if self._special_re is not None:
                last_end = None
                for m in self._special_re.finditer(buffer):
                    last_end = m.end()
                if last_end is not None:
                    yield from self.encode(buffer[:last_end])
                    buffer = buffer[last_end:]
            # 2) in the rest, every pre-token except the last is final (the regex is greedy
            #    and its only lookahead is (?!\S)), as long as it cannot touch a special token
            safe_end = len(buffer) - max(self._max_special_len - 1, 0)
            cut = 0
            for m in PAT_RE.finditer(buffer):
                if m.end() >= len(buffer) or m.end() > safe_end:
                    break
                cut = m.end()
            if cut:
                yield from self._encode_ordinary(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    # ------------------------------------------------------------------ decoding
    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
