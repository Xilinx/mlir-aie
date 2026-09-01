#!/usr/bin/env python3
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: ryzen_ai_npu2, xrt_python_bindings, peano
#
# RUN: %PEANO_INSTALL_DIR/bin/clang++ --target=aie2p-none-unknown-elf -O2 \
# RUN:   -std=c++20 -DNDEBUG -c %S/kernel.cc -o kernel.o
# RUN: %run_on_npu2% %python %S/test.py | FileCheck %s
#
# CHECK: PASS!

"""Six traced dispatches through aiex.configure share one trace buffer.

@main runs @dev_a and @dev_b twice each, so -aie-fuse-trace-buffers must give
every aiex.run its own slice of a single trace argument. The design is compiled
from a pre-written .mlir file because IRON builds one device with one runtime
sequence, and this needs three devices with five.
"""

import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.utils.trace import (
    TraceConfig,
    get_trace_buffer,
    get_trace_slices,
    parse_trace_slices,
)

HERE = Path(__file__).parent

# One (device, sequence) pair per aiex.run in @main, in dispatch order, with the
# runtime parameter that sequence writes.
RUNS = [
    ("dev_a", "seq_a1", 500),
    ("dev_b", "seq_b1", 700),
    ("dev_a", "seq_a2", 900),
    ("dev_a", "seq_a2", 900),
    ("dev_b", "seq_b2", 1100),
    ("dev_b", "seq_b2", 1100),
]

SLICE_BYTES = 8192
WORDS_PER_RUN = 4
EVENT_OF_DEVICE = {"dev_a": "INSTR_EVENT_0", "dev_b": "INSTR_EVENT_1"}


def count_events(events, name):
    return sum(1 for e in events if e.get("name") == name and e.get("ph") == "B")


def main():
    errors = []

    trace_config = TraceConfig(trace_size=len(RUNS) * SLICE_BYTES)
    design = iron.jit(
        HERE / "aie.mlir",
        full_elf=True,
        trace_config=trace_config,
        object_files=[str(Path.cwd() / "kernel.o")],
    )

    out = iron.zeros(len(RUNS) * WORDS_PER_RUN, dtype=np.int32, device="npu")
    design(out)
    out.to("cpu")

    # Each run writes its runtime parameter into its own window of the output,
    # which pins the dispatch order.
    expected = np.zeros(len(RUNS) * WORDS_PER_RUN, dtype=np.int32)
    for i, (_, _, rtp) in enumerate(RUNS):
        expected[i * WORDS_PER_RUN] = rtp
    if not np.array_equal(out.numpy(), expected):
        errors.append(f"output {out.numpy().tolist()} != {expected.tolist()}")

    mlir = Path(trace_config.physical_mlir_path).read_text()

    buffer_info = get_trace_buffer(mlir)
    if buffer_info["size"] != len(RUNS) * SLICE_BYTES:
        errors.append(f"fused trace buffer is {buffer_info['size']} bytes")

    # -aie-fuse-trace-buffers records one slice per aiex.run, in buffer order.
    slices = get_trace_slices(mlir)
    got = [(s["device"], s["sequence"], s["offset"], s["size"]) for s in slices]
    want = [
        (device, sequence, i * SLICE_BYTES, SLICE_BYTES)
        for i, (device, sequence, _) in enumerate(RUNS)
    ]
    if got != want:
        errors.append(f"slices {got} != {want}")

    # A slice decodes against the device that wrote it, so a slice must never
    # carry the other device's event.
    trace_buffer = trace_config.read_trace()
    populated = 0
    for info, events in parse_trace_slices(trace_buffer, mlir):
        mine = count_events(events, EVENT_OF_DEVICE[info["device"]])
        other = info["device"] == "dev_a" and "INSTR_EVENT_1" or "INSTR_EVENT_0"
        theirs = count_events(events, other)
        print(f"{info['device']}/{info['sequence']}@{info['offset']}: {mine} events")
        if theirs:
            errors.append(f"{info['device']} slice holds {theirs} {other}")
        if mine:
            populated += 1
    if populated == 0:
        errors.append("no slice received trace data")

    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        return 1
    print("PASS!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
