"""Generate a story with the trained model. Needs only torch, regex and einops (no GPU).

python inference.py --model-dir release --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
"""
import argparse
import json
import os
import time

import torch

from model import TransformerLM, generate
from tokenizer import Tokenizer

EOT = "<|endoftext|>"


def load_model(model_dir: str, device: str) -> TransformerLM:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    model = TransformerLM.from_config(config)
    state = torch.load(os.path.join(model_dir, "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser(description="Generate text from a trained TinyStories model")
    ap.add_argument("--model-dir", default="release")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-new-tokens", type=int, default=0, help="forbid <|endoftext|> before this many tokens (off by default)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    model = load_model(args.model_dir, args.device)
    tokenizer = Tokenizer.from_files(os.path.join(args.model_dir, "vocab.json"), os.path.join(args.model_dir, "merges.txt"), [EOT])

    start = time.perf_counter()
    text, n_new = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        min_new_tokens=args.min_new_tokens,
        generator=generator,
    )
    seconds = time.perf_counter() - start
    print(text)
    print(f"\n[generated {n_new} tokens in {seconds:.2f} s = {n_new / seconds:.1f} tokens/s on {args.device}]")


if __name__ == "__main__":
    main()
