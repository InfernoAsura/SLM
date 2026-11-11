from collections import Counter

import pytest

from tokenizer.bpe import find_chunk_boundaries, merge_pairs, train_bpe, train_bpe_from_counts

EOT = "<|endoftext|>"


def test_hand_worked_example_first_six_merges():
    # low x5, lower x2, widest x3, newest x6, split on spaces only.
    # Worked by hand: (s,t)=9 and (e,s)=9 tie -> the larger tuple (s,t) wins; then e+st=9,
    # o+w=7, l+ow=7, w+est=6, n+e=6 (ties broken by the larger pair).
    words = {"low": 5, "lower": 2, "widest": 3, "newest": 6}
    word_counts = {tuple(bytes([c]) for c in w.encode()): n for w, n in words.items()}
    merges = merge_pairs(word_counts, 6)
    assert merges == [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]


def test_tie_break_prefers_larger_tuple():
    assert max([(b"A", b"B"), (b"BA", b"A")]) == (b"BA", b"A")
    # "ab" and "ba" both appear once -> (b"b", b"a") is the larger tuple and must win
    merges = merge_pairs({(b"a", b"b"): 1, (b"b", b"a"): 1}, 1)
    assert merges == [(b"b", b"a")]


def test_no_token_crosses_story_boundary(tmp_path):
    # The two stories share no byte at all. If the text is not split on EOT first, the
    # special token's own bytes get merged (b"|>", b"text", ...) and the check below fails
    # (verified by running the same training without the split).
    story1 = "ab" * 20
    story2 = "cd" * 20
    p = tmp_path / "two.txt"
    p.write_text((story1 + EOT + story2 + EOT) * 10, encoding="utf-8")
    vocab, merges = train_bpe(str(p), 300, [EOT])
    s1, s2 = set(story1.encode()), set(story2.encode())
    assert not s1 & s2
    assert len(vocab) > 257  # some merges did happen
    for tok in list(vocab.values())[257:]:
        # every learned token comes from one story only (and never from the special token text)
        assert set(tok) <= s1 or set(tok) <= s2, tok
    # the special token is in the vocab exactly once and never merged into anything
    assert sum(1 for t in vocab.values() if t == EOT.encode()) == 1
    assert all(EOT.encode() not in a + b for a, b in merges)


def test_vocab_size_exact_and_ids_unique(tmp_path):
    text = ("The quick brown fox jumps over the lazy dog. " * 3 + EOT) * 50
    text += "Once upon a time there was a little girl named Lily." * 20
    p = tmp_path / "t.txt"
    p.write_text(text, encoding="utf-8")
    vocab, merges = train_bpe(str(p), 280, [EOT])
    assert len(vocab) == 280
    assert len(set(vocab.keys())) == 280
    assert sorted(vocab.keys()) == list(range(280))
    assert len(set(vocab.values())) == 280  # no two ids share a token
    assert len(merges) == 280 - 257
    assert vocab[256] == EOT.encode()


def test_parallel_matches_serial(tmp_path):
    stories = [f"Story {i}: Tom had {i} red balls and a dog called Max." for i in range(200)]
    p = tmp_path / "s.txt"
    p.write_text(EOT.join(stories), encoding="utf-8")
    serial = train_bpe(str(p), 300, [EOT], num_workers=1)
    parallel = train_bpe(str(p), 300, [EOT], num_workers=2)
    assert serial == parallel


def test_chunk_boundaries_land_on_special_token(tmp_path):
    p = tmp_path / "s.txt"
    p.write_bytes((("hello world " * 30) + EOT).encode() * 100)
    bounds = find_chunk_boundaries(str(p), 7, EOT.encode())
    data = p.read_bytes()
    assert bounds[0] == 0 and bounds[-1] == len(data)
    for b in bounds[1:-1]:
        assert data[b : b + len(EOT)] == EOT.encode()
    # limited to the first ~half of the file
    half = find_chunk_boundaries(str(p), 3, EOT.encode(), max_bytes=len(data) // 2)
    assert half[-1] <= len(data) // 2 + 400 and data[half[-1] : half[-1] + len(EOT)] == EOT.encode()


def test_special_tokens_are_not_counted():
    vocab, merges = train_bpe_from_counts(Counter({"ab": 3}), 258, [EOT])
    assert merges == [(b"a", b"b")]
