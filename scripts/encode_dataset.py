"""Encode a TinyStories text file into a uint16 .npy array of token ids (in parallel).

python scripts/encode_dataset.py --tokenizer-dir artifacts/tokenizer \
    --input data/TinyStoriesV2-GPT4-train.txt --output data/train.npy --workers 12
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import Tokenizer  # noqa: E402
from tokenizer.bpe import find_chunk_boundaries  # noqa: E402

EOT = "<|endoftext|>"
_TOK = None


def _init(tok_dir):
    global _TOK
    _TOK = Tokenizer.from_files(os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt"), [EOT])


def _encode_chunk(args):
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    ids = _TOK.encode(text)
    assert max(ids, default=0) < 2**16
    return np.asarray(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    n_bytes = os.path.getsize(args.input)
    bounds = find_chunk_boundaries(args.input, max(1, args.workers * 8), EOT.encode())
    jobs = [(args.input, s, e) for s, e in zip(bounds[:-1], bounds[1:])]
    t0 = time.perf_counter()
    if args.workers <= 1:
        _init(args.tokenizer_dir)
        parts = [_encode_chunk(j) for j in jobs]
    else:
        with Pool(args.workers, initializer=_init, initargs=(args.tokenizer_dir,)) as pool:
            parts = list(pool.imap(_encode_chunk, jobs))  # imap keeps the original order
    ids = np.concatenate(parts)
    elapsed = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, ids)
    stats = {
        "input": args.input,
        "output": args.output,
        "bytes": n_bytes,
        "tokens": int(len(ids)),
        "bytes_per_token": round(n_bytes / len(ids), 4),
        "seconds": round(elapsed, 1),
        "workers": args.workers,
        "throughput_MB_per_s_total": round(n_bytes / elapsed / 1e6, 2),
        "dtype": str(ids.dtype),
        "max_id": int(ids.max()),
    }
    with open(args.output + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
