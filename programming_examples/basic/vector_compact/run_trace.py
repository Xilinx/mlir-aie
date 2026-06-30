#!/usr/bin/env python3
# run_trace.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Run both stream-compaction kernels on the NPU with hardware tracing and
report the device-measured event0->event1 core-cycle cost of each.

Pipeline (per kernel, butterfly then scalar):

  1. Compile a trace-enabled @iron.jit design (vector_compact_trace.py) to
     xclbin + insts.  This also drops ``<name>.prj/input_with_addresses.mlir``,
     the routed/physical MLIR the trace parser needs.
  2. Load + run it on the NPU via NPUKernel + DefaultNPURuntime.run_test with a
     TraceConfig, which extracts the on-device trace buffer to a ``.txt`` file.
  3. parse.py turns (raw trace .txt + physical MLIR) into a Chrome-tracing JSON.
  4. get_cycles() reads the JSON's INSTR_EVENT_0 / INSTR_EVENT_1 markers and
     returns the cycle delta (the region-of-interest the kernel brackets with
     event0()/event1()).

Finally it prints the cycle counts + speedup and calls
visualize_compact_trace.py to render ``compact_trace.png``.

Usage:
    python3 run_trace.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel, TraceConfig
from aie.utils.trace.utils import get_cycles

import vector_compact_trace as vct

_HERE = Path(__file__).resolve().parent
_BUILD = _HERE / "build_trace"
_PARSE = (
    _HERE.parents[2] / "python" / "utils" / "trace" / "parse.py"
)  # mlir-aie/python/utils/trace/parse.py
_VISUALIZE = _HERE / "visualize_compact_trace.py"

_N = vct._N
_TRACE_SIZE = vct._TRACE_SIZE
_KERNEL_NAME = "MLIR_AIE"  # symbol embedded in the xclbin by @iron.jit

# Theoretical op counts documented in the kernel (for the plot annotations):
#   butterfly: ~31 vector ops/tile * 32 tiles  = ~992 vector ops
#   scalar:    ~1024 scalar compare+store      = ~1024 scalar stores
_THEORY = {
    "butterfly": (992, "vector ops"),
    "scalar": (1024, "scalar stores"),
}

# (label, external C symbol)
_KERNELS = [
    ("butterfly", "bf16_vector_compact"),
    ("scalar", "bf16_scalar_compact"),
]


def _compile(label, symbol):
    """Compile the trace-enabled design for one kernel symbol.

    Returns (xclbin_path, insts_path, physical_mlir_path).
    """
    _BUILD.mkdir(exist_ok=True)
    xclbin = _BUILD / f"{label}.xclbin"
    insts = _BUILD / f"{label}.insts.bin"
    prj_mlir = _BUILD / f"{label}.prj" / "input_with_addresses.mlir"

    print(f"[{label}] compiling trace design ({symbol}) ...")
    design = vct.build_trace_design(symbol, trace_size=_TRACE_SIZE)
    design.specialize().compile(xclbin_path=str(xclbin), inst_path=str(insts))

    if not prj_mlir.exists():
        # Be robust to a future .prj layout change.
        found = list(_BUILD.glob(f"{label}.prj/**/input_with_addresses.mlir"))
        if not found:
            raise FileNotFoundError(
                f"[{label}] could not find input_with_addresses.mlir under "
                f"{_BUILD / (label + '.prj')}"
            )
        prj_mlir = found[0]
    return xclbin, insts, prj_mlir


def _run_with_trace(label, xclbin, insts):
    """Run the design on the NPU with tracing on; return the trace .txt path."""
    trace_txt = _BUILD / f"{label}.trace.txt"

    trace_config = TraceConfig(
        trace_size=_TRACE_SIZE,
        trace_file=str(trace_txt),
        ddr_id=4,
        enable_ctrl_pkts=True,
    )
    kernel = NPUKernel(
        xclbin_path=str(xclbin),
        insts_path=str(insts),
        kernel_name=_KERNEL_NAME,
        trace_config=trace_config,
    )

    # Same input as vector_compact.py / vector_compact_trace.py.
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1.0, 1.0, size=(_N,)).astype(bfloat16)
    survivors = x_np[x_np >= 0.0]
    k = len(survivors)
    ref = np.zeros(_N, dtype=bfloat16)
    ref[:k] = survivors

    a = iron.tensor(x_np, dtype=bfloat16, device="npu")
    c = iron.zeros(_N, dtype=bfloat16, device="npu")

    print(f"[{label}] running on NPU with trace ...")
    res = DefaultNPURuntime.run_test(
        kernel,
        [a, c],
        {1: ref},
        verify=True,
        verbosity=0,
    )
    if res != 0:
        print(f"[{label}] WARNING: functional verification returned {res}")

    if not trace_txt.exists():
        raise FileNotFoundError(f"[{label}] trace file not produced: {trace_txt}")
    return trace_txt, k


def _parse_to_json(label, trace_txt, prj_mlir):
    """Run parse.py to turn the raw trace into a Chrome-tracing JSON."""
    trace_json = _BUILD / f"{label}.trace.json"
    cmd = [
        sys.executable,
        str(_PARSE),
        "--input",
        str(trace_txt),
        "--mlir",
        str(prj_mlir),
        "--output",
        str(trace_json),
    ]
    print(f"[{label}] parsing trace -> {trace_json.name}")
    subprocess.run(cmd, check=True)
    return trace_json


def main():
    results = {}  # label -> dict(cycles, json, survivors)
    survivor_count = None

    for label, symbol in _KERNELS:
        xclbin, insts, prj_mlir = _compile(label, symbol)
        trace_txt, k = _run_with_trace(label, xclbin, insts)
        survivor_count = k
        trace_json = _parse_to_json(label, trace_txt, prj_mlir)

        cycles = get_cycles(str(trace_json))
        results[label] = {
            "cycles": cycles,
            "json": str(trace_json),
            "survivors": k,
        }
        print(f"[{label}] event0->event1 = {cycles} cycles\n")

    bfly = results["butterfly"]["cycles"]
    scal = results["scalar"]["cycles"]
    speedup = (scal / bfly) if (bfly and np.isfinite(bfly) and bfly > 0) else float("nan")

    print("=" * 56)
    print(
        f"Stream compaction trace (N={_N}, bf16, "
        f"survivors={survivor_count}/{_N})"
    )
    print("=" * 56)
    print(f"butterfly kernel: {bfly} cycles  (event0->event1 on AIE core clock)")
    print(f"scalar    kernel: {scal} cycles")
    print(f"speedup vs scalar: {speedup:.1f}x")
    print("=" * 56)

    # Visualize: hand the per-kernel JSONs + measured cycles to the plotter.
    out_png = _HERE / "compact_trace.png"
    cmd = [
        sys.executable,
        str(_VISUALIZE),
        "--butterfly-json",
        results["butterfly"]["json"],
        "--scalar-json",
        results["scalar"]["json"],
        "--butterfly-cycles",
        str(bfly),
        "--scalar-cycles",
        str(scal),
        "--survivors",
        str(survivor_count),
        "--output",
        str(out_png),
    ]
    print("\nGenerating comparison plot ...")
    subprocess.run(cmd, check=True)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
