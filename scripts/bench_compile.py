"""Time training steps with and without torch.compile (same model, same batches).

python scripts/bench_compile.py --train-data data/train.npy --steps 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TransformerLM  # noqa: E402
from train import build_optimizer, build_scheduler, get_batch, load_tokens, make_scaler, train_step  # noqa: E402


def run(data, compile_model, steps, warmup, dtype, device):
    torch.manual_seed(0)
    model = TransformerLM(10_000, 256, 512, 4, 16, 1344).to(device)
    opt = build_optimizer(model, 1e-3)
    sched = build_scheduler(opt, 1e-3, 1e-4, 10, steps + warmup)
    scaler = make_scaler(device.type, dtype)
    m = torch.compile(model) if compile_model else model
    rng = np.random.default_rng(0)
    batches = [get_batch(data, 32, 256, device, rng) for _ in range(steps + warmup)]  # load data up front: time only the model
    t_compile = time.perf_counter()
    for i in range(warmup):
        train_step(m, *batches[i], opt, sched, scaler, dtype)
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t_compile
    t0 = time.perf_counter()
    for i in range(warmup, warmup + steps):
        train_step(m, *batches[i], opt, sched, scaler, dtype)
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / steps
    return per_step, warm_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/train.npy")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--out", default="runs/bench_compile.json")
    args = ap.parse_args()
    device = torch.device("cuda")
    data = load_tokens(args.train_data)
    eager, eager_warm = run(data, False, args.steps, args.warmup, args.dtype, device)
    comp, comp_warm = run(data, True, args.steps, args.warmup, args.dtype, device)
    res = {
        "gpu": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "steps_timed": args.steps,
        "eager_ms_per_step": round(eager * 1000, 2),
        "compiled_ms_per_step": round(comp * 1000, 2),
        "speedup": round(eager / comp, 3),
        "first_steps_seconds_eager": round(eager_warm, 1),
        "first_steps_seconds_compiled_incl_compilation": round(comp_warm, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
