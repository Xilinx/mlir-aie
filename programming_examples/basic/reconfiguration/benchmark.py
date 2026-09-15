#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Benchmark the reconfiguration approaches over a set of array sizes and write
# the raw per-iteration runtimes to a CSV.  Plotting is done separately by
# plot.py (which reads the CSV).
#
# Each (approach, size) is built and run through the Makefile; the C++ testbench
# prints a `runtimes_us:` line with one time per iteration.
#
# Usage:
#   python3 benchmark.py [--iters N] [--output benchmark.csv]

import argparse
import csv
import subprocess
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

# name -> (cols, rows, nops).  Program memory is ~192 + 4*nops bytes/core;
# the 16 KB limit is reached around nops=4000.
CONFIGS = {
    "small": (1, 1, 100),
    "medium": (4, 2, 2000),
    "large": (8, 4, 4000),
}

# label -> make target.
APPROACHES = [
    ("separate xclbins", "run_separate"),
    ("XRT runlist", "run_runlist"),
    ("load_pdis", "run_loadpdi"),
    ("blockwrites + empty reset", "run_blockwrites"),
    ("control packets + load_pdi overlay", "run_ctrlpkt"),
]


def run_case(target, cols, rows, nops, iters, switchboxes=0):
    """Build+run one approach; return the list of per-iteration runtimes (us)."""
    cmd = [
        "make",
        "-C",
        str(SRC_DIR),
        target,
        f"COLS={cols}",
        f"ROWS={rows}",
        f"NOPS={nops}",
        f"SWITCHBOXES={switchboxes}",
        f"ITERS={iters}",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    times = None
    for line in out.stdout.splitlines():
        if line.startswith("runtimes_us:"):
            times = [float(x) for x in line.split(":", 1)[1].split(",")]
    if times is None:
        raise RuntimeError(
            f"no runtimes from {target} ({cols}x{rows} nops={nops} "
            f"switchboxes={switchboxes}):\n{out.stdout}\n{out.stderr}"
        )
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--output", default=str(SRC_DIR / "benchmark.csv"))
    args = ap.parse_args()

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "cols", "rows", "nops", "approach", "iter", "runtime_us"])
        for cfg_name, (cols, rows, nops) in CONFIGS.items():
            for label, target in APPROACHES:
                times = run_case(target, cols, rows, nops, args.iters)
                for i, t in enumerate(times):
                    w.writerow([cfg_name, cols, rows, nops, label, i, f"{t:.3f}"])
                print(
                    f"{cfg_name:8s} {label:36s} "
                    f"min={min(times):8.1f} us  ({len(times)} iters)"
                )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
