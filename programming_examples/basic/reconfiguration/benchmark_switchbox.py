#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sweep switchbox-configuration size and record reconfiguration runtime for each
# approach, writing the raw per-iteration times to a CSV.  Plot with
# plot_switchbox.py.
#
# The design is a single active compute core with one outbound flow (its output
# drain), which verifies the configuration actually loads and runs.  Every other
# compute-tile switchbox is filled directly with stream-switch configuration
# (see reconfiguration.py `_fill_switchboxes`); the sweep parameter is how many
# of those switchboxes are filled, so the X axis is the number of switchboxes
# the configuration touches.  One line per reconfiguration approach.
#
# Usage:
#   python3 benchmark_switchbox.py [--iters N] [--output switchbox.csv]

import argparse
import csv

from benchmark import APPROACHES, SRC_DIR, run_case

# The single core's outbound drain routes through its own, the mem tile's and
# the shim's switchbox: three used switchboxes before any padding is added.
BASE_SWITCHBOXES = 3

# Filled padding switchboxes; a single-core design leaves 24 inner-column
# compute tiles free (outer columns 0 and 7 have no horizontal transit).
SWITCHBOX_VALUES = list(range(0, 25, 2))  # 13 points


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--output", default=str(SRC_DIR / "switchbox.csv"))
    args = ap.parse_args()

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["switchboxes", "used_switchboxes", "approach", "iter", "runtime_us"]
        )
        for label, target in APPROACHES:
            for pad in SWITCHBOX_VALUES:
                used = BASE_SWITCHBOXES + pad
                times = run_case(target, 1, 1, 0, args.iters, switchboxes=pad)
                for i, t in enumerate(times):
                    w.writerow([pad, used, label, i, f"{t:.3f}"])
                print(f"{label:36s} used={used:3d} min={min(times):8.1f} us")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

