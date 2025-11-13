#!/usr/bin/env python3
# ===- test.py ------------------------------------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# ===-----------------------------------------------------------------------===#
#
# REQUIRES: ryzen_ai, chess
#

# Build the test
# RUN: xchesscc_wrapper aie2 -I %aietools/include -c %S/vector_scalar_mul.cc -o vector_scalar_mul.o
# RUN: %python aiecc.py --no-aiesim --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.bin %S/aie.mlir

# Run the test
# RUN: %run_on_npu1% %python %S/test.py --xclbin final.xclbin --instr insts.bin --kernel MLIR_AIE --trace-sz 8192 --mlir %S/aie.mlir | FileCheck %s
# CHECK: PASS!

import argparse
import numpy as np
import sys
from pathlib import Path
import pyxrt as xrt
from aie.utils.parse_trace import parse_trace


# Buffer sizes
IN1_SIZE = 16384  # bytes
IN2_SIZE = 4  # bytes
OUT_SIZE = 16384  # bytes

DATATYPE_IN1 = np.int32
DATATYPE_IN2 = np.int32
DATATYPE_OUT = np.int32

IN1_VOLUME = IN1_SIZE // np.dtype(DATATYPE_IN1).itemsize
IN2_VOLUME = IN2_SIZE // np.dtype(DATATYPE_IN2).itemsize
OUT_VOLUME = OUT_SIZE // np.dtype(DATATYPE_OUT).itemsize


def initialize_bufIn1(size):
    """Initialize Input buffer 1"""
    rng = np.random.default_rng(seed=42)
    return rng.integers(1, 100, size=size, dtype=DATATYPE_IN1)


def initialize_bufIn2(size):
    """Initialize Input buffer 2"""
    buf = np.zeros(size, dtype=DATATYPE_IN2)
    buf[0] = 3  # scaleFactor
    return buf


def initialize_bufOut(size):
    """Initialize Output buffer"""
    return np.zeros(size, dtype=DATATYPE_OUT)


def verify_vector_scalar_mul(bufIn1, bufIn2, bufOut, size, verbosity=0):
    """Functional correctness verifier"""
    errors = 0

    for i in range(size):
        ref = bufIn1[i] * bufIn2[0]
        test = bufOut[i]
        if test != ref:
            if verbosity >= 1:
                print(f"Error in output {test} != {ref}")
            errors += 1
        else:
            if verbosity >= 1:
                print(f"Correct output {test} == {ref}")

    return errors


def load_instr_binary(instr_file):
    """Load instruction binary file"""
    try:
        with open(instr_file, "rb") as f:
            instr_data = f.read()
        # Convert to uint32 array
        instr_v = np.frombuffer(instr_data, dtype=np.uint32)
        return instr_v
    except FileNotFoundError:
        print(f"ERROR: Instruction file '{instr_file}' not found.")
        sys.exit(1)


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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--xclbin", required=True, help="Path to XCLBIN file")
    parser.add_argument(
        "--instr", required=True, help="Path to instruction binary file"
    )
    parser.add_argument("--kernel", default="MLIR_AIE", help="Kernel name")
    parser.add_argument(
        "--verbosity", "-v", type=int, default=0, help="Verbosity level"
    )
    parser.add_argument(
        "--trace-sz", type=int, default=8192, help="Trace buffer size in bytes"
    )
    parser.add_argument("--mlir", help="MLIR source file for trace parsing")

    return parser.parse_args()


def main():
    args = parse_args()

    assert args.trace_sz > 0, "Trace size must be greater than 0"

    if args.verbosity >= 1:
        print("=" * 80)
        print(f"XCLBIN: {args.xclbin}")
        print(f"Instruction file: {args.instr}")
        print(f"Kernel: {args.kernel}")
        print(
            f"IN1_VOLUME: {IN1_VOLUME}, IN2_VOLUME: {IN2_VOLUME}, OUT_VOLUME: {OUT_VOLUME}"
        )
        print("=" * 80)

    # Load instruction sequence
    instr_v = load_instr_binary(args.instr)
    if args.verbosity >= 1:
        print(f"Sequence instr count: {len(instr_v)}")

    # Start the XRT context and load the kernel
    if args.verbosity >= 1:
        print("Loading device and kernel...")

    device = xrt.device(0)
    xclbin = xrt.xclbin(args.xclbin)
    device.register_xclbin(xclbin)

    # Get kernel name from xclbin
    xkernels = xclbin.get_kernels()
    xkernel = None
    for k in xkernels:
        if k.get_name() == args.kernel:
            xkernel = k
            break

    if xkernel is None:
        print(f"ERROR: Kernel '{args.kernel}' not found in xclbin")
        sys.exit(1)

    kernel = xrt.kernel(
        device, xclbin.get_uuid(), args.kernel, xrt.kernel.cu_access_mode.exclusive
    )

    if args.verbosity >= 1:
        print(f"Kernel loaded: {args.kernel}")

    # Set up buffer objects
    bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(1))
    bo_in1 = xrt.bo(device, IN1_SIZE, xrt.bo.host_only, kernel.group_id(3))
    bo_in2 = xrt.bo(device, IN2_SIZE, xrt.bo.host_only, kernel.group_id(4))
    bo_out = xrt.bo(device, OUT_SIZE, xrt.bo.host_only, kernel.group_id(5))
    bo_extra = xrt.bo(device, 8, xrt.bo.host_only, kernel.group_id(6))

    # Workaround: allocate trace buffer 4x size
    tmp_trace_size = args.trace_sz * 4
    bo_trace = xrt.bo(device, tmp_trace_size, xrt.bo.host_only, kernel.group_id(7))

    if args.verbosity >= 1:
        print("Writing data into buffer objects...")

    # Map buffers and initialize
    # Note: pyxrt write() expects bytes
    bo_instr.write(instr_v.tobytes(), 0)

    bufIn1 = initialize_bufIn1(IN1_VOLUME)
    bufIn2 = initialize_bufIn2(IN2_VOLUME)
    bufOut = initialize_bufOut(OUT_VOLUME)

    bo_in1.write(bufIn1.tobytes(), 0)
    bo_in2.write(bufIn2.tobytes(), 0)
    bo_out.write(bufOut.tobytes(), 0)

    # Initialize trace buffer with zeros
    trace_init = np.zeros(tmp_trace_size, dtype=np.uint8)
    bo_trace.write(trace_init.tobytes(), 0)

    # Sync host to device
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in1.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    bo_trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Run kernel
    if args.verbosity >= 1:
        print("Running Kernel...")

    opcode = 3
    run = kernel(
        opcode, bo_instr, len(instr_v), bo_in1, bo_in2, bo_out, bo_extra, bo_trace
    )
    run.wait()

    # Sync device to host
    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    bo_trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Read output buffer
    bufOut_bytes = bo_out.read(OUT_SIZE, 0)
    bufOut = np.frombuffer(bufOut_bytes, dtype=DATATYPE_OUT)

    # Verify results
    if args.verbosity >= 1:
        print("Verifying results...")

    errors = verify_vector_scalar_mul(
        bufIn1, bufIn2, bufOut, IN1_VOLUME, args.verbosity
    )

    # Write trace to file
    trace_data_bytes = bo_trace.read(args.trace_sz, 0)

    # Convert to uint32 array for proper formatting
    trace_buffer = np.frombuffer(trace_data_bytes, dtype=np.uint32)

    if args.verbosity >= 1:
        print(f"Trace buffer shape: {trace_buffer.shape}")
        print(f"Trace buffer dtype: {trace_buffer.dtype}")

    trace_events = generate_trace_json(
        trace_buffer, args.mlir, verbosity=args.verbosity
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

    if args.verbosity >= 1:
        print(f"INSTR_EVENT_0 count: {instr_event_0_count}")
        print(f"INSTR_EVENT_1 count: {instr_event_1_count}")

    # Verify expected counts. The kernel is expected to generate 4 of each event.
    if instr_event_0_count != 4:
        print(f"ERROR: Expected 4 INSTR_EVENT_0 events, found {instr_event_0_count}")
        errors += 1
    if instr_event_1_count != 4:
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
    sys.exit(main())
