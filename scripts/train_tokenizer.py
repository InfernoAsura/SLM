"""Train the 10k BPE tokenizer on TinyStories and save vocab.json / merges.txt.

python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --out-dir artifacts/tokenizer --workers 8
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import save_tokenizer  # noqa: E402
from tokenizer.bpe import pretokenize_file, train_bpe_from_counts  # noqa: E402

EOT = "<|endoftext|>"


class PeakMemory:
    """Samples the resident memory of this process plus all its children (the pool workers)."""

    def __init__(self, interval=0.2):
        self.interval, self.peak, self._stop = interval, 0, False

    def _run(self):
        import psutil

        me = psutil.Process()
        while not self._stop:
            try:
                rss = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
                self.peak = max(self.peak, rss)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        try:
            import psutil  # noqa: F401

            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        except ImportError:
            self._t = None
        return self

    def __exit__(self, *exc):
        self._stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max-bytes", type=int, default=None, help="only use the first N bytes of the file")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with PeakMemory() as mem:
        t0 = time.perf_counter()
        counts = pretokenize_file(args.input, [EOT], num_workers=args.workers, max_bytes=args.max_bytes)
        t1 = time.perf_counter()
        print(f"pre-tokenisation: {t1 - t0:.1f}s, {len(counts):,} unique pre-tokens", flush=True)
        vocab, merges = train_bpe_from_counts(counts, args.vocab_size, [EOT])
        t2 = time.perf_counter()
        print(f"merging: {t2 - t1:.1f}s, {len(merges):,} merges", flush=True)

    save_tokenizer(vocab, merges, os.path.join(args.out_dir, "vocab.json"), os.path.join(args.out_dir, "merges.txt"))
    longest = max(vocab.values(), key=len)
    stats = {
        "input": args.input,
        "max_bytes": args.max_bytes,
        "workers": args.workers,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "unique_pretokens": len(counts),
        "pretokenize_seconds": round(t1 - t0, 2),
        "merge_seconds": round(t2 - t1, 2),
        "total_seconds": round(t2 - t0, 2),
        "peak_memory_mb": round(mem.peak / 2**20, 1),
        "longest_token": longest.decode("utf-8", errors="replace"),
        "longest_token_bytes": len(longest),
        "top10_longest": [t.decode("utf-8", errors="replace") for t in sorted(vocab.values(), key=len, reverse=True)[:10]],
        "first_20_merges": [(a + b).decode("utf-8", errors="replace") for a, b in merges[:20]],
    }
    with open(os.path.join(args.out_dir, "train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
