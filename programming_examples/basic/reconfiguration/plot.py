#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Plot the reconfiguration benchmark CSV (from benchmark.py) as a grouped bar
# chart on a black background.  For each (config, approach) the first `warmup`
# iterations are dropped and the median of the rest is plotted, with min/max
# whiskers.  The x-axis annotates both the array shape and the program-memory
# footprint, since reconfiguration cost scales with both.
#
# Usage:
#   python3 plot.py [--csv benchmark.csv] [--output benchmark.png] [--warmup W]

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent

# Approaches to plot, in bar order.
PLOT_APPROACHES = [
    "separate xclbins",
    "XRT runlist",
    "load_pdis",
    "blockwrites + empty reset",
    "control packets + load_pdi overlay",
]


def prog_mem_kb(nops):
    # Per-core program memory: the core body plus `nops` event instructions,
    # empirically ~192 + 4*nops bytes (the 16 KB core limit is hit near nops=4000).
    return (192 + 4 * nops) / 1024.0


def load(csv_path, warmup):
    runtimes = defaultdict(list)  # (config, approach) -> [(iter, runtime_us)]
    shape = {}                    # config -> (cols, rows, nops)
    order = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cfg, approach = row["config"], row["approach"]
            runtimes[(cfg, approach)].append(
                (int(row["iter"]), float(row["runtime_us"]))
            )
            shape[cfg] = (int(row["cols"]), int(row["rows"]), int(row["nops"]))
            if cfg not in order:
                order.append(cfg)

    summary = {}  # (config, approach) -> (median, min, max) over steady state
    for key, iters in runtimes.items():
        vals = [t for i, t in sorted(iters) if i >= warmup] or [t for _, t in iters]
        summary[key] = (statistics.median(vals), min(vals), max(vals))
    return order, shape, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(SRC_DIR / "benchmark.csv"))
    ap.add_argument("--output", default=str(SRC_DIR / "benchmark.png"))
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    configs, shape, summary = load(args.csv, args.warmup)

    labels = []
    for c in configs:
        cols, rows, nops = shape[c]
        cores = cols * rows
        pc = prog_mem_kb(nops)
        labels.append(
            f"{c}\n{cols}\u00d7{rows} = {cores} cores\n"
            f"{pc:.1f} KB/core, {pc * cores:.0f} KB total"
        )

    n_groups = len(configs)
    n_bars = len(PLOT_APPROACHES)
    width = 0.8 / n_bars
    x = range(n_groups)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="black")
    ax.set_facecolor("black")

    for b, approach in enumerate(PLOT_APPROACHES):
        meds, los, his = [], [], []
        for cfg in configs:
            med, lo, hi = summary[(cfg, approach)]
            meds.append(med)
            los.append(med - lo)
            his.append(hi - med)
        offs = [xi + (b - (n_bars - 1) / 2) * width for xi in x]
        bars = ax.bar(offs, meds, width, label=approach, yerr=[los, his],
                      capsize=2, ecolor="0.7")
        for rect, med, hi in zip(bars, meds, his):
            ax.annotate(f"{med:.0f}",
                        (rect.get_x() + rect.get_width() / 2, med + hi),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, rotation=90,
                        color="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("reconfigure + run time per iteration (us)")
    ax.set_title("NPU reconfiguration cost vs array size and program memory")
    ax.legend(fontsize=8, facecolor="black", edgecolor="0.5", labelcolor="white")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
