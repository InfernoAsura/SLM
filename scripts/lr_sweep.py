"""Short learning-rate pilot runs (each a normal train.py run with warm-up + cosine).

python scripts/lr_sweep.py --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 --steps 300
"""
import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out-root", default="runs/lr_sweep")
    ap.add_argument("--train-data", default="data/train.npy")
    ap.add_argument("--val-data", default="data/valid.npy")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to train.py")
    args = ap.parse_args()
    results = {}
    for lr in args.lrs:
        out = os.path.join(args.out_root, f"lr_{lr:g}")
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--train-data", args.train_data, "--val-data", args.val_data,
               "--out-dir", out, "--max-steps", str(args.steps), "--lr", str(lr), "--eval-every", "50", "--print-every", "50",
               "--ckpt-every", "1000000", *args.extra]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        rows = list(csv.DictReader(open(os.path.join(out, "log.csv"))))
        vals = [float(r["val_loss"]) for r in rows if r["val_loss"]]
        trains = [float(r["train_loss"]) for r in rows if r["train_loss"]]
        results[f"{lr:g}"] = {
            "final_val_loss": vals[-1] if vals else None,
            "last_train_loss": trains[-1] if trains else None,
            "steps_completed": int(rows[-1]["step"]),
            "max_grad_norm": max(float(r["grad_norm"]) for r in rows if r["grad_norm"]),
        }
        print(lr, results[f"{lr:g}"], flush=True)
    with open(os.path.join(args.out_root, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
