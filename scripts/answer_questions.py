"""Reproduce every number used in the writeup for the warm-up questions and Q1-Q8.

python scripts/answer_questions.py --tokenizer-dir artifacts/tokenizer --out report/results/analysis.json
(add --skip-profile to skip the slow cProfile runs)
"""
import argparse
import cProfile
import io
import json
import math
import os
import pstats
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import RoPE, TransformerLM  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402
from tokenizer.bpe import train_bpe  # noqa: E402

EOT = "<|endoftext|>"


def warmups():
    out = {}
    s = "a" + chr(0) + "b"
    out["chr0_repr"] = repr(chr(0))
    out["chr0_in_string_repr"] = repr(s)
    out["chr0_in_string_print_len"] = len(s)
    sizes = {}
    for t in ["hello", "Once upon a time", "नमस्ते", "😀", "héllo"]:
        sizes[t] = {"utf8": len(t.encode("utf-8")), "utf16": len(t.encode("utf-16")), "utf32": len(t.encode("utf-32"))}
    out["encoding_sizes_bytes_incl_BOM_for_16_32"] = sizes
    sizes_nobom = {t: {"utf16le": len(t.encode("utf-16-le")), "utf32le": len(t.encode("utf-32-le"))} for t in sizes}
    out["encoding_sizes_no_BOM"] = sizes_nobom
    word = "héllo"
    try:
        "".join(bytes([b]).decode("utf-8") for b in word.encode("utf-8"))
        out["bytewise_decode"] = "no error"
    except UnicodeDecodeError as e:
        out["bytewise_decode"] = f"{word!r} -> {word.encode('utf-8')!r} -> UnicodeDecodeError: {e}"
    try:
        b"\xc3\x28".decode("utf-8")
        out["invalid_two_bytes"] = "decoded?!"
    except UnicodeDecodeError as e:
        out["invalid_two_bytes"] = f"b'\\xc3\\x28' -> {e}"
    return out


def profile_bpe(path, vocab_size=10_000, max_bytes=None):
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    train_bpe(path, vocab_size, [EOT], num_workers=1, max_bytes=max_bytes)
    prof.disable()
    total = time.perf_counter() - t0
    st = pstats.Stats(prof)
    cum = {}
    for (fname, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if fname.endswith("bpe.py") and func in ("pretokenize_file", "merge_pairs", "_merge_word", "train_bpe_from_counts"):
            cum[func] = cum.get(func, 0) + ct
        if "builtins.max" in func:
            cum["builtins.max (finding the best pair)"] = cum.get("builtins.max (finding the best pair)", 0) + ct
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(8)
    t1 = time.perf_counter()
    train_bpe(path, vocab_size, [EOT], num_workers=1, max_bytes=max_bytes)
    no_prof = time.perf_counter() - t1
    return {
        "file": path,
        "max_bytes": max_bytes,
        "profiled_total_s": round(total, 2),
        "unprofiled_total_s": round(no_prof, 2),
        "cumulative_s": {k: round(v, 2) for k, v in cum.items()},
        "top_by_tottime": s.getvalue().splitlines()[:40],
    }


def tokenizer_stats(tok_dir, valid_path):
    tok = Tokenizer.from_files(os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt"), [EOT])
    text = open(valid_path, encoding="utf-8").read()
    stories = [s.strip() for s in text.split(EOT) if s.strip()]
    random.seed(0)
    sample = random.sample(stories, 10)
    nb = sum(len(s.encode()) for s in sample)
    nt = sum(len(tok.encode(s)) for s in sample)
    for s in sample:
        assert tok.decode(tok.encode(s)) == s
    # single-process throughput on 5 MB with a fresh (cold-cache) tokenizer
    tok_cold = Tokenizer.from_files(os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt"), [EOT])
    chunk = text[:5_000_000]
    t0 = time.perf_counter()
    ids = tok_cold.encode(chunk)
    dt = time.perf_counter() - t0
    vocab = tok.vocab
    longest = max(vocab.values(), key=len)
    return {
        "compression_10_random_stories_bytes_per_token": round(nb / nt, 3),
        "sample_bytes": nb,
        "sample_tokens": nt,
        "single_process_throughput_bytes_per_s": round(len(chunk.encode()) / dt),
        "single_process_tokens": len(ids),
        "longest_token": longest.decode("utf-8", "replace"),
        "longest_token_len": len(longest),
        "example_encoding": [vocab[i].decode("utf-8", "replace") for i in tok.encode("Once upon a time, there was a little girl named Lily.")],
    }


def q3_sqrt_dk():
    torch.manual_seed(0)
    d_k = 32
    q = torch.randn(8, d_k)
    k = torch.randn(8, d_k)
    raw = q @ k.T
    scaled = raw / math.sqrt(d_k)
    p_raw, p_scaled = torch.softmax(raw, -1), torch.softmax(scaled, -1)

    def ent(p):
        return float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean())

    # gradient of softmax wrt scores ~ p(1-p): tiny when p is near 0 or 1
    return {
        "score_std_raw": round(float(raw.std()), 3),
        "score_std_scaled": round(float(scaled.std()), 3),
        "row0_softmax_raw": [round(x, 4) for x in p_raw[0].tolist()],
        "row0_softmax_scaled": [round(x, 4) for x in p_scaled[0].tolist()],
        "mean_max_prob_raw": round(float(p_raw.max(-1).values.mean()), 4),
        "mean_max_prob_scaled": round(float(p_scaled.max(-1).values.mean()), 4),
        "mean_entropy_raw_nats": round(ent(p_raw), 4),
        "mean_entropy_scaled_nats": round(ent(p_scaled), 4),
        "max_entropy_nats_8_keys": round(math.log(8), 4),
        "mean_softmax_jacobian_diag_raw": round(float((p_raw * (1 - p_raw)).mean()), 5),
        "mean_softmax_jacobian_diag_scaled": round(float((p_scaled * (1 - p_scaled)).mean()), 5),
    }


def q4_rope():
    torch.manual_seed(0)
    d = 32
    rope = RoPE(10_000.0, d, 256)
    q, k = torch.randn(d), torch.randn(d)
    Q = rope(q.expand(256, d).clone())
    K = rope(k.expand(256, d).clone())
    pairs = [(3, 0), (10, 7), (100, 97), (250, 247), (5, 0), (50, 45), (7, 7), (200, 200), (0, 3), (97, 100)]
    return {"unrotated_dot": round(float(q @ k), 4), "dots": {f"({i},{j}) i-j={i - j}": round(float(Q[i] @ K[j]), 4) for i, j in pairs}}


def lr_warmup():
    res = {}
    for lr in [1, 10, 100, 1000]:
        torch.manual_seed(0)
        w = torch.nn.Parameter(5 * torch.randn(10, 10))
        opt = torch.optim.SGD([w], lr=lr)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = (w**2).mean()
            losses.append(float(loss))
            loss.backward()
            opt.step()
        res[str(lr)] = [float(f"{x:.4g}") for x in losses]
    # gradient of mean(w^2) is 2w/100 = w/50, so one step multiplies w by (1 - lr/50)
    res["factor_per_step_(1-lr/50)"] = {str(lr): 1 - lr / 50 for lr in [1, 10, 100, 1000]}
    return res


def q5_params():
    V, T, d, L, h, dff = 10_000, 256, 512, 4, 16, 1344
    m = TransformerLM(V, T, d, L, h, dff)
    parts = {
        "token_embedding": m.token_emb.weight.numel(),
        "per_layer_attention_qkvo": sum(p.numel() for n, p in m.blocks[0].attn.named_parameters()),
        "per_layer_ffn_swiglu": sum(p.numel() for p in m.blocks[0].ffn.parameters()),
        "per_layer_rmsnorms": m.blocks[0].norm1.weight.numel() + m.blocks[0].norm2.weight.numel(),
        "per_layer_total": sum(p.numel() for p in m.blocks[0].parameters()),
        "all_layers": sum(p.numel() for p in m.blocks.parameters()),
        "final_norm": m.final_norm.weight.numel(),
        "lm_head": m.lm_head.weight.numel(),
    }
    total = sum(p.numel() for p in m.parameters())
    parts["total"] = total
    parts["hand_total"] = V * d + L * (4 * d * d + 3 * d * dff + 2 * d) + d + d * V
    parts["embedding_plus_output_share"] = round((parts["token_embedding"] + parts["lm_head"]) / total, 4)
    return parts


def flops(V, T, d, L, dff, n_ffn_mats=3):
    per_layer = {
        "q,k,v,o projections (4 x 2*T*d*d)": 4 * 2 * T * d * d,
        "attention scores QK^T (2*T*T*d)": 2 * T * T * d,
        "attention-weighted values AV (2*T*T*d)": 2 * T * T * d,
        f"feed-forward ({n_ffn_mats} x 2*T*d*d_ff)": n_ffn_mats * 2 * T * d * dff,
    }
    layer_total = sum(per_layer.values())
    head = 2 * T * d * V
    total = L * layer_total + head
    groups = {
        "attention projections": L * per_layer["q,k,v,o projections (4 x 2*T*d*d)"],
        "attention scores + AV": L * 2 * (2 * T * T * d),
        "feed-forward": L * per_layer[f"feed-forward ({n_ffn_mats} x 2*T*d*d_ff)"],
        "output layer (lm_head)": head,
    }
    return {
        "per_layer_flops": per_layer,
        "per_layer_total": layer_total,
        "lm_head (2*T*d*V)": head,
        "total_forward_flops": total,
        "share": {k: round(v / total, 4) for k, v in groups.items()},
        "attention_scores_share_within_a_layer": round(2 * (2 * T * T * d) / layer_total, 4),
    }


def q8_memory():
    n = 22_696_448
    b = 4
    return {
        "params_MB": round(n * b / 1e6, 1),
        "grads_MB": round(n * b / 1e6, 1),
        "adamw_m_and_v_MB": round(2 * n * b / 1e6, 1),
        "total_MB_no_activations": round(4 * n * b / 1e6, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", default="artifacts/tokenizer")
    ap.add_argument("--valid", default="data/TinyStoriesV2-GPT4-valid.txt")
    ap.add_argument("--train", default="data/TinyStoriesV2-GPT4-train.txt")
    ap.add_argument("--out", default="report/results/analysis.json")
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args()
    res = {
        "warmups": warmups(),
        "tokenizer": tokenizer_stats(args.tokenizer_dir, args.valid),
        "q3": q3_sqrt_dk(),
        "q4": q4_rope(),
        "lr_warmup_sgd": lr_warmup(),
        "q5": q5_params(),
        "q6": flops(10_000, 256, 512, 4, 1344, 3),
        "q7_gpt2_small_swiglu_dff2048": flops(50_257, 1024, 768, 12, 2048, 3),
        "q7_gpt2_small_gelu_dff3072": flops(50_257, 1024, 768, 12, 3072, 2),
        "q8": q8_memory(),
    }
    if not args.skip_profile:
        res["q1_profile_valid"] = profile_bpe(args.valid)
        res["q1_profile_train_first_200MB"] = profile_bpe(args.train, max_bytes=200_000_000)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("q1")}, indent=1, ensure_ascii=False)[:6000])
    for k in ("q1_profile_valid", "q1_profile_train_first_200MB"):
        if k in res:
            print(k, {kk: vv for kk, vv in res[k].items() if kk != "top_by_tottime"})


if __name__ == "__main__":
    main()
