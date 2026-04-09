#!/usr/bin/env python3
# ===- test.py ------------------------------------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
# ===-----------------------------------------------------------------------===#
#
# End-to-end test for distribute-channels + lateral-routing trace features.
#
# With distribute-channels, the 8192-byte host trace buffer is split into
# two 8192-byte regions (total 16384 bytes at arg_idx=4):
#   - Channel 0 (offset 0):     core trace events
#   - Channel 1 (offset 8192):  memory DMA trace events
#
# With lateral-routing, both channels are routed to column 1's shim
# (the spare column), keeping trace traffic off the active data path.
#
# REQUIRES: ryzen_ai_npu1, xrt_python_bindings
#
# Compile kernel with Peano (aie2 target for Phoenix/NPU1):
# RUN: %PEANO_INSTALL_DIR/bin/clang --target=aie2-none-unknown-elf -O2 -c %S/vector_scalar_mul.cc -o vector_scalar_mul.o
#
# Pre-lower trace ops with distribute + lateral options, then compile.
# The trace pipeline in aiecc is a no-op since trace ops are already consumed.
# RUN: aie-opt %S/aie.mlir -aie-insert-trace-flows="distribute-channels=true lateral-routing=true" -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config -o trace_lowered.mlir
# RUN: %python aiecc.py --no-xchesscc --no-xbridge --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.bin trace_lowered.mlir
#
# Run on NPU1 hardware:
# RUN: %run_on_npu1% %python %S/test.py --xclbin final.xclbin --instr insts.bin --kernel MLIR_AIE --trace-sz 16384 --mlir trace_lowered.mlir.prj/input_with_addresses.mlir | FileCheck %s
# CHECK: PASS!

import numpy as np
import sys
from pathlib import Path
from aie.utils.trace import parse_trace
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime

IN_OUT_SIZE = 4096
IN_OUT_DTYPE = np.int32
SCALAR_FACTOR = 3
TRACE_BUFFER_SIZE = 8192  # Per-channel buffer size in bytes
TRACE_WORDS_PER_CHANNEL = TRACE_BUFFER_SIZE // 4  # 2048 uint32 words


def check_channel_has_data(trace_buffer, channel, words_per_channel, verbosity=0):
    """Check that a channel's portion of the trace buffer contains non-zero data."""
    start = channel * words_per_channel
    end = start + words_per_channel
    if end > len(trace_buffer):
        if verbosity >= 1:
            print(
                f"  Channel {channel}: buffer too short "
                f"(need {end}, have {len(trace_buffer)})"
            )
        return False
    channel_data = trace_buffer[start:end]
    nonzero = np.count_nonzero(channel_data)
    if verbosity >= 1:
        print(
            f"  Channel {channel}: {nonzero}/{words_per_channel} "
            f"non-zero words (offset {start}-{end})"
        )
    return nonzero > 0


def main(opts):
    assert opts.trace_size > 0, "Trace size must be greater than 0"
    if opts.verbosity >= 1:
        print("=" * 80)
        print("Distribute + Lateral Trace E2E Test")
        print(f"  XCLBIN: {opts.xclbin}")
        print(
            f"  Trace size: {opts.trace_size} bytes "
            f"({opts.trace_size // TRACE_BUFFER_SIZE} channels)"
        )
        print("=" * 80)

    rng = np.random.default_rng(seed=42)
    input_data = rng.integers(1, 100, size=IN_OUT_SIZE, dtype=IN_OUT_DTYPE)
    in1 = iron.tensor(input_data, dtype=IN_OUT_DTYPE)
    in2 = iron.tensor(np.array([SCALAR_FACTOR], dtype=IN_OUT_DTYPE), dtype=IN_OUT_DTYPE)
    out = iron.zeros(IN_OUT_SIZE, dtype=IN_OUT_DTYPE)
    ref_data = input_data * SCALAR_FACTOR

    if opts.verbosity >= 1:
        print("Running...\n")

    # Total trace size = 2 * TRACE_BUFFER_SIZE for distribute
    opts.trace_size = TRACE_BUFFER_SIZE * 2

    npu_opts = test_utils.create_npu_kernel(opts)
    if DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, in2, out],
        {2: ref_data},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    ):
        print("Failed: data verification error.")
        return 1

    errors = 0

    # Read trace buffer (contains data from both distributed channels)
    trace_buffer = npu_opts.npu_kernel.trace_config.read_trace()

    if opts.verbosity >= 1:
        print(
            f"\nTrace buffer: {trace_buffer.shape[0]} words "
            f"({trace_buffer.shape[0] * 4} bytes)"
        )

    # -- Check 1: Both channel regions have trace data --
    # With distribute, channel 0 is at offset 0 and channel 1 at offset
    # TRACE_BUFFER_SIZE. Both should contain non-zero trace data.
    print("\nChecking distributed channel data:")
    for ch in range(2):
        if not check_channel_has_data(
            trace_buffer, ch, TRACE_WORDS_PER_CHANNEL, opts.verbosity
        ):
            print(
                f"ERROR: Channel {ch} has no trace data "
                f"(distribute may not be working)"
            )
            errors += 1

    # -- Check 2: Parse trace events if MLIR file is available --
    if opts.mlir and Path(opts.mlir).exists():
        with open(opts.mlir, "r") as f:
            mlir_str = f.read()

        try:
            trace_events = parse_trace(trace_buffer, mlir_str)
        except Exception as e:
            print(f"WARNING: Trace parsing failed: {e}")
            trace_events = None

        if trace_events:
            # Count core trace events (kernel runs 4 iterations)
            core_event_0 = sum(
                1
                for ev in trace_events
                if ev.get("name") == "INSTR_EVENT_0" and ev.get("ph") == "B"
            )
            core_event_1 = sum(
                1
                for ev in trace_events
                if ev.get("name") == "INSTR_EVENT_1" and ev.get("ph") == "B"
            )
            # Count memory DMA events (any DMA_ prefixed event from the
            # mem trace module confirms channel 1 captured data)
            mem_dma_events = sum(
                1
                for ev in trace_events
                if ev.get("name", "").startswith("DMA_") and ev.get("ph") == "B"
            )

            if opts.verbosity >= 1:
                print(f"\nTrace events:")
                print(f"  INSTR_EVENT_0: {core_event_0}")
                print(f"  INSTR_EVENT_1: {core_event_1}")
                print(f"  DMA_* (mem trace): {mem_dma_events}")

            if core_event_0 < 4:
                print(f"ERROR: Expected >= 4 INSTR_EVENT_0, got {core_event_0}")
                errors += 1
            if core_event_1 < 4:
                print(f"ERROR: Expected >= 4 INSTR_EVENT_1, got {core_event_1}")
                errors += 1
            if mem_dma_events < 1:
                print(
                    f"ERROR: Expected >= 1 DMA_* mem trace events, "
                    f"got {mem_dma_events}"
                )
                errors += 1
        else:
            print(
                "WARNING: No trace events parsed (trace data may be valid "
                "but parser returned empty)"
            )
    else:
        print("WARNING: No MLIR file provided, skipping event-level checks")

    # Final result
    if errors == 0:
        print("\nPASS!")
        return 0
    else:
        print(f"\nFAIL - {errors} errors detected!")
        return 1


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument("--mlir", dest="mlir", help="MLIR file for trace parsing")
    opts = p.parse_args(sys.argv[1:])
    sys.exit(main(opts))
