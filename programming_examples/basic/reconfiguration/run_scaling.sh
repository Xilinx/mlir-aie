#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Reproduce every figure in this example.  Run from the example directory on an
# NPU2 (Strix) host, after `source ~/setup_buildenv.sh`.
set -euo pipefail
cd "$(dirname "$0")"

# Default comparison: grouped bar chart over small/medium/large array sizes,
# five approaches (benchmark.png).
python3 benchmark.py --iters 12          # -> benchmark.csv
python3 plot.py                          # benchmark.csv -> benchmark.png

# Scaling study 1: reconfiguration time vs core program memory.
# Tiny 1x1 array (constant, minimal switchbox), NOPS swept 0..4000 (17 points),
# all five approaches.
python3 benchmark_progmem.py --iters 12  # -> progmem.csv
python3 plot_progmem.py                  # progmem.csv -> progmem.png

# Scaling study 2: reconfiguration time vs switchbox configuration.
# A single active core (one outbound flow), SWITCHBOXES swept 0..24 unused
# switchboxes filled directly with stream-switch configuration; one line per
# approach, X axis the number of used switchboxes.
python3 benchmark_switchbox.py --iters 12  # -> switchbox.csv
python3 plot_switchbox.py                  # switchbox.csv -> switchbox.png
