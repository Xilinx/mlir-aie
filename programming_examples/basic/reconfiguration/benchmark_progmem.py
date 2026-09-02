#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sweep core program-memory size (via NOPS) and record reconfiguration runtime
# for every approach, writing the raw per-iteration times to a CSV.  The array
# is kept tiny (1x1, no filled switchboxes) so switchbox config stays constant
# and program memory is the only thing that grows.  Plot with plot_progmem.py.
#
# Usage:
#   python3 benchmark_progmem.py [--iters N] [--output progmem.csv]

import argparse
import csv

from benchmark import APPROACHES, SRC_DIR, run_case

# ~192 + 4*nops bytes/core; the 16 KB core limit is reached near nops=4000.
NOPS_VALUES = list(range(0, 4001, 250))  # 17 points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--cols", type=int, default=1)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--output", default=str(SRC_DIR / "progmem.csv"))
    args = ap.parse_args()

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nops", "progmem_bytes", "approach", "iter", "runtime_us"])
        for nops in NOPS_VALUES:
            progmem = 192 + 4 * nops
            for label, target in APPROACHES:
                times = run_case(
                    target, args.cols, args.rows, nops, args.iters, switchboxes=0
                )
                for i, t in enumerate(times):
                    w.writerow([nops, progmem, label, i, f"{t:.3f}"])
                print(f"nops={nops:5d} {label:36s} min={min(times):8.1f} us")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
