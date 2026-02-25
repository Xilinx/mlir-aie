#!/usr/bin/env python3
# ===- test.py ------------------------------------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
#
# ===-----------------------------------------------------------------------===#
#
# REQUIRES: ryzen_ai_npu1, xrt_python_bindings
#

# Build the test
# RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/vector_scalar_mul.cc -o vector_scalar_mul.o
# RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.bin %S/aie.mlir

# Run the test
# RUN: %run_on_npu1% %python %S/test.py --xclbin final.xclbin --instr insts.bin --kernel MLIR_AIE --trace-sz 8192 --mlir %S/aie.mlir | FileCheck %s
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


def generate_trace_json(trace_buffer, mlir_file, verbosity=0):
    """Generate trace json from trace buffer using parse_trace API"""
    try:
        # Read MLIR file
        if not Path(mlir_file).exists():
            print(
                f"WARNING: MLIR file '{mlir_file}' not found. Skipping trace.json generation."
            )
            return False

        with open(mlir_file, "r") as f:
            mlir_str = f.read()

        if verbosity >= 1:
            print(f"Parsing trace with MLIR configuration from {mlir_file}...")

        # Parse trace buffer using new API
        return parse_trace(trace_buffer, mlir_str, debug=(verbosity >= 2))

    except Exception as e:
        print(f"ERROR: Failed to generate trace JSON: {e}")
        if verbosity >= 2:
            import traceback

            traceback.print_exc()
        return False


def main(opts):
    assert opts.trace_size > 0, "Trace size must be greater than 0"
    if opts.verbosity >= 1:
        print("=" * 80)
        print(f"XCLBIN: {opts.xclbin}")
        print(f"Instruction file: {opts.instr}")
        print(f"Kernel: {opts.kernel}")
        print("=" * 80)

    rng = np.random.default_rng(seed=42)
    input_data = rng.integers(1, 100, size=IN_OUT_SIZE, dtype=IN_OUT_DTYPE)
    in1 = iron.tensor(input_data, dtype=IN_OUT_DTYPE)
    in2 = iron.tensor([SCALAR_FACTOR], dtype=IN_OUT_DTYPE)
    out = iron.zeros(IN_OUT_SIZE, dtype=IN_OUT_DTYPE)
    ref_data = input_data * SCALAR_FACTOR

    # Start the XRT context and load the kernel
    if opts.verbosity >= 1:
        print("Running...\n")

    opts.trace_size = IN_OUT_SIZE * 4

    npu_opts = test_utils.create_npu_kernel(opts)
    if DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, in2, out],
        {2: ref_data},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    ):
        print("Failed.")
        return 1
    errors = 0

    # Read trace from file
    trace_buffer = npu_opts.npu_kernel.trace_config.read_trace()

    if opts.verbosity >= 1:
        print(f"Trace buffer shape: {trace_buffer.shape}")
        print(f"Trace buffer dtype: {trace_buffer.dtype}")

    trace_events = generate_trace_json(
        trace_buffer, opts.mlir, verbosity=opts.verbosity
    )

    if not trace_events:
        print("ERROR: Failed to generate trace events (empty or False returned).")
        errors += 1
        instr_event_0_count = 0
        instr_event_1_count = 0
    else:
        instr_event_0_count = sum(
            1
            for event in trace_events
            if event.get("name") == "INSTR_EVENT_0" and event.get("ph") == "B"
        )
        instr_event_1_count = sum(
            1
            for event in trace_events
            if event.get("name") == "INSTR_EVENT_1" and event.get("ph") == "B"
        )

    if opts.verbosity >= 1:
        print(f"INSTR_EVENT_0 count: {instr_event_0_count}")
        print(f"INSTR_EVENT_1 count: {instr_event_1_count}")

    # Verify expected counts. The kernel is expected to generate 4 of each event.
    if instr_event_0_count < 4:
        print(f"ERROR: Expected 4 INSTR_EVENT_0 events, found {instr_event_0_count}")
        errors += 1
    if instr_event_1_count < 4:
        print(f"ERROR: Expected 4 INSTR_EVENT_1 events, found {instr_event_1_count}")
        errors += 1

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
