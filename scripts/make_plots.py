"""Make every figure for the writeup from the CSV logs in runs/.

python scripts/make_plots.py --runs runs --out report/figures
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": TEXT, "legend.frameon": False, "figure.dpi": 150,
})


def read(path):
    rows = list(csv.DictReader(open(path)))

    def col(name):
        xs, ys = [], []
        for r in rows:
            if r[name] not in ("", None):
                xs.append(int(r["step"]))
                ys.append(float(r[name]))
        return np.array(xs), np.array(ys)

    t = {int(r["step"]): float(r["elapsed_s"]) for r in rows if r["elapsed_s"]}
    return col, t


def smooth(y, k=25):
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([y[: k - 1] * np.nan, out])


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def single_batch(runs, out):
    col, _ = read(os.path.join(runs, "single_batch", "log.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x, y = col("train_loss")
    ax.plot(x, y, color=PALETTE[0], label="training loss (the one batch)")
    x, y = col("val_loss")
    ax.plot(x, y, color=PALETTE[1], marker="o", markersize=4, label="validation loss (fixed held-out batches)")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Single-batch check: the model memorises one batch")
    ax.legend(loc="center right")
    save(fig, out, "single_batch.png")


def resume(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, (name, label, ls) in enumerate([("resume_full", "uninterrupted run", "-"), ("resume_split", "stopped at 300, resumed", "--")]):
        col, _ = read(os.path.join(runs, name, "log.csv"))
        x, y = col("train_loss")
        axes[0].plot(x, smooth(y, 10), color=PALETTE[i], ls=ls, label=label)
        x, y = col("val_loss")
        axes[1].plot(x, y, color=PALETTE[i], ls=ls, marker="o", markersize=4, label=label)
    for ax, t in zip(axes, ["Training loss (10-step mean)", "Validation loss"]):
        ax.set_ylim(2.2, 4.5)
        ax.axvline(300, color=MUTED, lw=1, ls=":")
        ax.text(305, 2.3, "stop / resume", color=MUTED, va="bottom", fontsize=8)
        ax.set_title(t)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.legend()
    save(fig, out, "resume_check.png")


def grad_norm(runs, out):
    col, _ = read(os.path.join(runs, "main", "log.csv"))
    x, g = col("grad_norm")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    m = x <= 100
    axes[0].plot(x[m], g[m], color=PALETTE[0])
    axes[0].set_title("Gradient norm, first 100 steps")
    axes[0].axhline(1.0, color=MUTED, lw=1, ls=":")
    axes[0].text(100, 1.02, "clip at 1.0", ha="right", va="bottom", color=MUTED, fontsize=8)
    finite = np.isfinite(g)
    axes[1].plot(x[finite], g[finite], color=PALETTE[0], lw=1)
    axes[1].set_title("Gradient norm, whole run (before clipping)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("global L2 norm")
    save(fig, out, "grad_norm.png")


def lr_sweep(runs, out):
    summary = json.load(open(os.path.join(runs, "lr_sweep", "summary.json")))
    lrs = sorted(summary, key=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, lr in enumerate(lrs):
        col, _ = read(os.path.join(runs, "lr_sweep", f"lr_{lr}", "log.csv"))
        x, y = col("train_loss")
        fin = np.isfinite(y)
        lab = f"lr {float(lr):g}" + (f" (NaN at step {int(x[~fin][0])})" if not fin.all() else "")
        if fin.all():
            axes[0].plot(x, smooth(y, 10), color=PALETTE[i], label=lab)
        else:  # too short to smooth: raw values, an x where the loss became NaN
            axes[0].plot(x[fin], y[fin], color=PALETTE[i], lw=1.5, label=lab)
            axes[0].plot(x[fin][-1], min(y[fin][-1], 9.9), "x", color=PALETTE[i], markersize=9, mew=2)
        xv, yv = col("val_loss")
        axes[1].plot(xv, yv, color=PALETTE[i], marker="o", markersize=4, label=lab)
    axes[0].set_title("Pilot runs, fp16: training loss (10-step mean)")
    axes[0].set_ylim(2.4, 10)
    axes[0].text(300, 9.8, "lr 0.1 rises to 19.5 (off the chart)", ha="right", va="top", color=MUTED, fontsize=8)
    axes[1].set_title("Pilot runs, fp16: validation loss")
    axes[1].set_ylim(2.4, 9.6)
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
    axes[1].legend(loc="upper right", fontsize=8)
    save(fig, out, "lr_sweep.png")

    # fp16 vs fp32 for the rates that blew up
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, lr in enumerate(["0.03", "0.1"]):
        for prec, ls in [("lr_sweep", "-"), ("lr_sweep_fp32", "--")]:
            p = os.path.join(runs, prec, f"lr_{lr}", "log.csv")
            if not os.path.exists(p):
                continue
            col, _ = read(p)
            x, y = col("train_loss")
            fin = np.isfinite(y)
            name = "fp16" if prec == "lr_sweep" else "fp32"
            ax.plot(x[fin], y[fin], color=PALETTE[i + 5], ls=ls, lw=1.5, label=f"lr {lr} {name}" + (" -> NaN" if not fin.all() else ""))
            if not fin.all():
                ax.plot(x[fin][-1], y[fin][-1], "x", color=PALETTE[i + 5], markersize=9, mew=2)
    ax.axhline(np.log(10_000), color=MUTED, lw=1, ls=":")
    ax.text(300, np.log(10_000) + 0.3, "random guess ln(10000)", ha="right", color=MUTED, fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title("Rates that blew up: fp16 (NaN) vs fp32 (spike, then recovers)")
    ax.legend(fontsize=8)
    save(fig, out, "lr_fp16_vs_fp32.png")


def main_run(runs, out):
    col, tmap = read(os.path.join(runs, "main", "log.csv"))
    x, y = col("train_loss")
    xv, yv = col("val_loss")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    # The laptop went into Windows Modern Standby once during the run (one step took 631 s).
    # Remove any pause longer than 60 s so the time axis shows active training time.
    steps = sorted(tmap)
    active, removed, prev = {}, 0.0, 0.0
    for s in steps:
        dt = tmap[s] - prev
        if dt > 60 and s > 1:  # step 1 includes the ~30 s torch.compile, which is real work
            removed += dt - 0.08
        active[s] = tmap[s] - removed
        prev = tmap[s]
    t = np.array([active[s] for s in x]) / 60
    tv = np.array([active[s] for s in xv]) / 60
    tlabel = "active time, minutes (incl. 32 s compile" + (f"; {removed:.0f} s standby removed)" if removed else ")")
    for ax, xs, xsv, lab in [(axes[0], x, xv, "step"), (axes[1], t, tv, tlabel)]:
        ax.plot(xs, y, color=PALETTE[0], alpha=0.25, lw=1)
        ax.plot(xs, smooth(y, 50), color=PALETTE[0], label="training loss (50-step mean)")
        ax.plot(xsv, yv, color=PALETTE[1], marker="o", markersize=4, label="validation loss")
        ax.set_xlabel(lab)
        ax.set_ylabel("loss")
        ax.set_ylim(1.2, 5)
        ax.legend()
    axes[0].set_title("Main run vs steps")
    axes[1].set_title("Main run vs time")
    save(fig, out, "main_run.png")

    xl, lr = col("lr")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(xl, lr, color=PALETTE[2])
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule of the main run")
    save(fig, out, "lr_schedule.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in (single_batch, resume, lr_sweep, grad_norm, main_run):
        try:
            f(args.runs, args.out)
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)


if __name__ == "__main__":
    main()
