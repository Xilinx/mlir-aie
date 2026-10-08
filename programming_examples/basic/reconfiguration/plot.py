#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent

MODE_LABELS = {
    "separate-dispatch": "separate dispatches",
    "load-pdi": "single-dispatch load_pdi",
    "expand-load-pdis": "single-dispatch write32s",
    "control-packets": "single-dispatch ctrlpackets",
}
SEABORN_MAKO = (
    "#382a54",
    "#395d9c",
    "#3497a9",
    "#60ceac",
)


def load(csv_path, case, x_column):
    runtimes = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"case", "mode", x_column, "runtime_us"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["case"] != case:
                continue
            runtimes[row["mode"]][int(row[x_column])].append(float(row["runtime_us"]))
    return runtimes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(SRC_DIR / "benchmark.csv"))
    parser.add_argument("--x", choices=("nops", "switchboxes"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.x == "nops":
        case = "progmem"
        xlabel = "event(0) padding instructions per core"
        title = "Reconfiguration time vs core program-memory padding"
    else:
        case = "switchboxes"
        xlabel = "configured switchboxes"
        title = "Reconfiguration time vs switchbox configuration"
    runtimes = load(args.csv, case, args.x)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
    ax.set_facecolor("black")

    for index, (mode, label) in enumerate(MODE_LABELS.items()):
        series = runtimes[mode]
        xs = sorted(series)
        ys = [statistics.median(series[x]) for x in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=label,
            color=SEABORN_MAKO[index],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("reconfiguration time (us)")
    ax.set_title(title)
    ax.legend(facecolor="black", edgecolor="0.5", labelcolor="white")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
