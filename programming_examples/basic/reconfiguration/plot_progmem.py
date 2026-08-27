#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Line plot of reconfiguration runtime vs core program memory, from the CSV
# written by benchmark_progmem.py.  One line per approach; median over the
# steady-state iterations, with a min/max band.  Black background.
#
# Usage:
#   python3 plot_progmem.py [--csv progmem.csv] [--output progmem.png] [--warmup W]

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent

APPROACH_ORDER = [
    "separate xclbins",
    "XRT runlist",
    "load_pdis",
    "blockwrites + empty reset",
    "control packets + load_pdi overlay",
]


def load(csv_path):
    # data[approach][progmem_kb] = [(iter, runtime_us), ...]
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            kb = int(row["progmem_bytes"]) / 1024.0
            data[row["approach"]][kb].append(
                (int(row["iter"]), float(row["runtime_us"]))
            )
    return data


def summarize(iters, warmup):
    vals = [t for i, t in sorted(iters) if i >= warmup] or [t for _, t in iters]
    return statistics.median(vals), min(vals), max(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(SRC_DIR / "progmem.csv"))
    ap.add_argument("--output", default=str(SRC_DIR / "progmem.png"))
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    data = load(args.csv)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")

    for approach in APPROACH_ORDER:
        if approach not in data:
            continue
        xs = sorted(data[approach])
        meds, los, his = [], [], []
        for x in xs:
            m, lo, hi = summarize(data[approach][x], args.warmup)
            meds.append(m)
            los.append(lo)
            his.append(hi)
        (line,) = ax.plot(xs, meds, marker="o", ms=3, label=approach)
        ax.fill_between(xs, los, his, alpha=0.15, color=line.get_color())

    ax.set_xlabel("core program memory (KB)")
    ax.set_ylabel("reconfigure + run time per iteration (us)")
    ax.set_title(
        "Reconfiguration cost vs core program memory\n(1x1 array, no filled switchboxes)"
    )
    ax.legend(fontsize=8, facecolor="black", edgecolor="0.5", labelcolor="white")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
