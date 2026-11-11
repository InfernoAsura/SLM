import io

import pytest

from tokenizer import Tokenizer, load_tokenizer_files, save_tokenizer, train_bpe

EOT = "<|endoftext|>"


@pytest.fixture
def tiny():
    vocab = {0: b" ", 1: b"a", 2: b"c", 3: b"e", 4: b"h", 5: b"t", 6: b"th", 7: b" c", 8: b" a", 9: b"the", 10: b" at"}
    merges = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    return Tokenizer(vocab, merges)


def test_hand_worked_encoding(tiny):
    # "the" -> t h e -> th e -> the = [9]; " cat" -> " c" a t = [7, 1, 5];
    # " ate" -> " a" t e -> " at" e = [10, 3]
    assert tiny.encode("the cat ate") == [9, 7, 1, 5, 10, 3]
    assert tiny.decode([9, 7, 1, 5, 10, 3]) == "the cat ate"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    text = (
        "Once upon a time, there was a little girl named Lily. She liked to play outside.\n"
        "One day she saw a big red ball. \"Wow!\" she said. नमस्ते दुनिया 😀 123 4567\n"
    )
    p = tmp_path_factory.mktemp("bpe") / "train.txt"
    p.write_text((text + EOT) * 30, encoding="utf-8")
    vocab, merges = train_bpe(str(p), 400, [EOT])
    return Tokenizer(vocab, merges, [EOT])


@pytest.mark.parametrize(
    "s",
    [
        "",
        "Hello, world! It's a nice day.",
        "नमस्ते दुनिया, आप कैसे हैं?",
        "emoji 😀🎉👍🏽 and 🇮🇳",
        "  multiple   spaces \n\n\tand tabs  ",
        "Unseen wordz like xyzzy qwerty",
        f"one{EOT}two {EOT}{EOT} three",
    ],
)
def test_roundtrip(trained, s):
    assert trained.decode(trained.encode(s)) == s


def test_special_token_is_one_id_and_never_merged(trained):
    eot_id = trained.special_ids[EOT]
    ids = trained.encode(f"girl{EOT}girl")
    assert ids.count(eot_id) == 1
    k = ids.index(eot_id)
    # the text on either side is encoded independently
    assert ids[:k] == trained.encode("girl") and ids[k + 1 :] == trained.encode("girl")
    assert trained.encode(EOT) == [eot_id]


def test_overlapping_special_tokens_longest_wins():
    vocab = {i: bytes([i]) for i in range(256)}
    double = EOT + EOT
    tok = Tokenizer(vocab, [], [EOT, double])
    ids = tok.encode(f"a{double}b{EOT}c")
    assert tok.special_ids[double] in ids
    assert ids.count(tok.special_ids[EOT]) == 1
    assert tok.decode(ids) == f"a{double}b{EOT}c"


def test_special_token_added_to_end_if_missing():
    vocab = {i: bytes([i]) for i in range(256)}
    tok = Tokenizer(vocab, [], ["<|special|>"])
    assert tok.special_ids["<|special|>"] == 256
    assert tok.vocab_size == 257


def test_decode_invalid_utf8_does_not_crash(trained):
    # byte 0xE0 alone is the start of a 3-byte character -> replacement char
    bad_id = trained.token_to_id[bytes([0xE0])]
    assert trained.decode([bad_id]) == "�"


def test_encode_iterable_matches_encode(trained, tmp_path):
    text = ("Lily  saw a dog.\n\nHe said \"hi\" \n" + EOT + " नमस्ते 😀\n   end\n") * 40
    p = tmp_path / "f.txt"
    p.write_text(text, encoding="utf-8")
    with open(p, encoding="utf-8") as f:
        streamed = list(trained.encode_iterable(f))
    assert streamed == trained.encode(text)
    # tiny chunks that cut words, whitespace runs and the special token in half
    chunks = [text[i : i + 5] for i in range(0, len(text), 5)]
    assert list(trained.encode_iterable(chunks)) == trained.encode(text)


def test_save_and_load_roundtrip(trained, tmp_path):
    vocab = {i: t for i, t in trained.vocab.items()}
    merges = sorted(trained.merge_ranks, key=trained.merge_ranks.get)
    save_tokenizer(vocab, merges, tmp_path / "v.json", tmp_path / "m.txt")
    v2, m2 = load_tokenizer_files(tmp_path / "v.json", tmp_path / "m.txt")
    assert v2 == vocab and m2 == merges
    tok2 = Tokenizer.from_files(tmp_path / "v.json", tmp_path / "m.txt", [EOT])
    s = "Once upon a time 😀 with a token that has a space: ' a'"
    assert tok2.encode(s) == trained.encode(s)
