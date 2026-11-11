"""Generate the sample stories for the writeup (Part 7).

python scripts/make_stories.py --model-dir release --out report/stories.txt
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import EOT, load_model  # noqa: E402
from model import generate  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

SETTINGS = [  # (temperature, top_p, seed)
    (0.7, 0.9, 1),
    (0.7, 0.9, 2),
    (1.0, 0.95, 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="release")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="report/stories.txt")
    args = ap.parse_args()
    model = load_model(args.model_dir, args.device)
    tok = Tokenizer.from_files(os.path.join(args.model_dir, "vocab.json"), os.path.join(args.model_dir, "merges.txt"), [EOT])

    results = {"required": [], "natural_lengths": {}}
    lines = []
    for temp, top_p, seed in SETTINGS:
        g = torch.Generator(device=args.device).manual_seed(seed)
        # <|endoftext|> is blocked until 256 new tokens exist, so every story is >= 256 tokens
        text, n = generate(model, tok, args.prompt, max_new_tokens=256, temperature=temp, top_p=top_p, min_new_tokens=256, generator=g)
        # what the model does when it may stop by itself (same seed)
        g = torch.Generator(device=args.device).manual_seed(seed)
        natural_text, natural_n = generate(model, tok, args.prompt, max_new_tokens=512, temperature=temp, top_p=top_p, generator=g)
        results["required"].append({"temperature": temp, "top_p": top_p, "seed": seed, "new_tokens": n, "text": text,
                                    "natural_stop_tokens": natural_n})
        lines += [f"=== temperature {temp}, top-p {top_p}, seed {seed}: {n} new tokens "
                  f"(<|endoftext|> blocked until 256; left alone the model stops after {natural_n} tokens) ===", text, ""]

    for temp, top_p in [(0.7, 0.9), (1.0, 0.95)]:
        lens = []
        for seed in range(20):
            g = torch.Generator(device=args.device).manual_seed(1000 + seed)
            _, n = generate(model, tok, args.prompt, max_new_tokens=512, temperature=temp, top_p=top_p, generator=g)
            lens.append(n)
        results["natural_lengths"][f"T={temp},p={top_p}"] = {"lengths": lens, "median": statistics.median(lens),
                                                             "min": min(lens), "max": max(lens)}
    # greedy, for reference
    text, n = generate(model, tok, args.prompt, max_new_tokens=256, temperature=0)
    results["greedy"] = {"new_tokens": n, "text": text}
    lines += [f"=== greedy (temperature 0): {n} new tokens ===", text, ""]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n".join(lines))
    print(json.dumps(results["natural_lengths"], indent=1))


if __name__ == "__main__":
    main()
