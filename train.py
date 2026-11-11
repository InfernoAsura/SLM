"""Train the TinyStories language model.

Every model / optimizer setting can be given on the command line or in a JSON config file
(--config); command-line values override the file. Works on a local GPU, Colab or Kaggle
(fp16 autocast + GradScaler, or plain fp32 with --dtype fp32), and on CPU for debugging.

Example (main run):
  python train.py --train-data data/train.npy --val-data data/valid.npy \
      --out-dir runs/main --max-steps 5000 --lr 3e-3 --compile
Resume:
  python train.py ... --out-dir runs/main --resume runs/main/ckpt.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from model import TransformerLM
from train import (
    batch_from_starts,
    build_optimizer,
    build_scheduler,
    evaluate,
    fixed_eval_starts,
    get_batch,
    load_checkpoint,
    load_tokens,
    make_scaler,
    save_checkpoint,
    train_step,
    unwrap,
)

LOG_FIELDS = ["step", "train_loss", "val_loss", "lr", "grad_norm", "elapsed_s", "tokens_seen", "loss_scale", "skipped"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file with any of the options below (CLI overrides it)")
    # data / output
    p.add_argument("--train-data", default="data/train.npy")
    p.add_argument("--val-data", default="data/valid.npy")
    p.add_argument("--out-dir", default="runs/main")
    # model
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    # optimisation
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-3, help="peak learning rate (alpha_max)")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="alpha_min = ratio * alpha_max")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="fp16 = autocast + GradScaler (CUDA only)")
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    # evaluation / logging / checkpoints
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=50, help="number of FIXED validation batches")
    p.add_argument("--eval-seed", type=int, default=1234)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--stop-at-step", type=int, default=None, help="stop early (simulates an interruption); the schedule still targets --max-steps")
    p.add_argument("--single-batch", action="store_true", help="train on one fixed batch over and over (sanity check)")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config) as f:
            cfg = json.load(f)
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return parser.parse_args(argv)


def read_log(path: str, upto_step: int) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if int(r["step"]) <= upto_step]


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if device.type != "cuda" and args.dtype == "fp16":
        print("fp16 autocast needs CUDA; falling back to fp32")
        args.dtype = "fp32"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)  # batch sampling; saved in checkpoints
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # only affects fp32 matmuls on Ampere+; T4 ignores it
        torch.backends.cudnn.allow_tf32 = True

    train_data = load_tokens(args.train_data)
    val_data = load_tokens(args.val_data)
    print(f"train tokens {len(train_data):,} | val tokens {len(val_data):,} | device {device} | dtype {args.dtype}")

    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)

    optimizer = build_optimizer(model, args.lr, args.weight_decay, (args.beta1, args.beta2))
    warmup = round(args.warmup_frac * args.max_steps)
    # the step index runs 0 .. max_steps-1; the last step uses exactly alpha_min
    scheduler = build_scheduler(optimizer, args.lr, args.lr * args.min_lr_ratio, warmup, args.max_steps - 1)
    scaler = make_scaler(device.type, args.dtype)

    start_step, elapsed_before = 0, 0.0
    log_path = os.path.join(args.out_dir, "log.csv")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, rng=rng, map_location=device)
        start_step = state["step"]
        elapsed_before = state["extra"].get("elapsed_s", 0.0)
        rows = read_log(log_path, start_step)  # drop rows written after the checkpoint
        print(f"resumed from {args.resume} at step {start_step} (elapsed {elapsed_before:.1f}s)")
    else:
        rows = []
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    log_file = open(log_path, "a", newline="")
    logger = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)

    fwd_model = torch.compile(model) if args.compile else model

    # fixed validation set: the same batches at every evaluation and in every run
    eval_starts = fixed_eval_starts(len(val_data), args.eval_batches, args.batch_size, args.context_length, args.eval_seed)
    val_batches = [batch_from_starts(val_data, s, args.context_length, device) for s in eval_starts]

    single = get_batch(train_data, args.batch_size, args.context_length, device, np.random.default_rng(args.seed)) if args.single_batch else None

    def ckpt(step, name="ckpt.pt"):
        save_checkpoint(os.path.join(args.out_dir, name), model, optimizer, scheduler, scaler, step, rng, extra={"elapsed_s": elapsed(), "args": vars(args)})

    t0 = time.perf_counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - t0

    def log(row):
        logger.writerow(row)
        log_file.flush()

    if start_step == 0:
        v = evaluate(fwd_model, val_batches, args.dtype)
        log({"step": 0, "val_loss": f"{v:.5f}", "lr": 0.0, "elapsed_s": 0.0, "tokens_seen": 0})
        print(f"step 0 | val {v:.4f}")

    end_step = min(args.max_steps, args.stop_at_step or args.max_steps)
    tokens_per_step = args.batch_size * args.context_length
    last_val = None
    for step in range(start_step, end_step):
        lr = optimizer.param_groups[0]["lr"]  # the rate used for this update
        x, y = single if single is not None else get_batch(train_data, args.batch_size, args.context_length, device, rng)
        out = train_step(fwd_model, x, y, optimizer, scheduler, scaler, args.dtype, args.grad_clip)
        done = step + 1
        row = {
            "step": done,
            "train_loss": f"{out['loss']:.5f}",
            "lr": f"{lr:.6e}",
            "grad_norm": f"{out['grad_norm']:.5f}",
            "elapsed_s": f"{elapsed():.2f}",
            "tokens_seen": done * tokens_per_step,
            "loss_scale": out["scale"],
            "skipped": int(out["skipped"]),
            "val_loss": "",
        }
        if not math.isfinite(out["loss"]):
            log(row)
            print(f"step {done}: loss is {out['loss']} -> stopping (diverged)")
            break
        if done % args.eval_every == 0 or done == args.max_steps:
            last_val = evaluate(fwd_model, val_batches, args.dtype)
            row["val_loss"] = f"{last_val:.5f}"
        log(row)
        if done % args.print_every == 0 or done == 1 or row["val_loss"]:
            vl = f" | val {last_val:.4f}" if row["val_loss"] else ""
            print(f"step {done:5d} | loss {out['loss']:.4f}{vl} | lr {lr:.2e} | gnorm {out['grad_norm']:.3f} | {elapsed():.0f}s", flush=True)
        if done % args.ckpt_every == 0 and done < end_step:
            ckpt(done)

    final_step = done if end_step > start_step else start_step
    ckpt(final_step)
    log_file.close()
    if final_step >= args.max_steps:
        torch.save(unwrap(model).state_dict(), os.path.join(args.out_dir, "model.pt"))
        summary = {
            "final_step": final_step,
            "final_val_loss": last_val,
            "final_val_perplexity": math.exp(last_val) if last_val is not None else None,
            "train_seconds": elapsed(),
            "tokens_seen": final_step * tokens_per_step,
            "parameters": n_params,
        }
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
