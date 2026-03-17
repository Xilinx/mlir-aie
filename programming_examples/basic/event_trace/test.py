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

import argparse
import numpy as np
import time
import sys
from pathlib import Path

try:
    import pyxrt as xrt
except ImportError:
    print("ERROR: Could not import pyxrt. Please install XRT Python bindings.")
    sys.exit(1)

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


def write_out_trace(trace_buffer, trace_file):
    """Write trace buffer to file"""
    try:
        # Filter out zero values and write as hex strings
        out_str = "\n".join(f"{i:08x}" for i in trace_buffer if i != 0)
        with open(trace_file, "w") as f:
            f.write(out_str)
        print(f"Trace written to {trace_file}")
    except Exception as e:
        print(f"ERROR: Could not write trace file: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="XRT Test Wrapper (Python)")

    parser.add_argument("--xclbin", required=True, help="Path to XCLBIN file")
    parser.add_argument(
        "--instr", required=True, help="Path to instruction binary file"
    )
    parser.add_argument("--kernel", default="MLIR_AIE", help="Kernel name")
    parser.add_argument(
        "--verbosity", "-v", type=int, default=0, help="Verbosity level"
    )
    parser.add_argument(
        "--verify", action="store_true", default=True, help="Verify results"
    )
    parser.add_argument(
        "--no-verify", dest="verify", action="store_false", help="Skip verification"
    )
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations")
    parser.add_argument(
        "--warmup", type=int, default=0, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--trace-sz", type=int, default=0, help="Trace buffer size in bytes"
    )
    parser.add_argument("--trace-file", default="trace.txt", help="Trace output file")

    return parser.parse_args()


def main():
    args = parse_args()

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
    bo_ctrlpkts = xrt.bo(device, 8, xrt.bo.host_only, kernel.group_id(6))

    # Workaround: allocate trace buffer (small if not used, 4x size if used)
    tmp_trace_size = args.trace_sz * 4 if args.trace_sz > 0 else 1
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

    if args.trace_sz > 0:
        # Initialize trace buffer with zeros
        trace_init = np.zeros(tmp_trace_size, dtype=np.uint8)
        bo_trace.write(trace_init.tobytes(), 0)

    # Sync host to device
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in1.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    if args.trace_sz > 0:
        bo_trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Initialize run configs
    num_iter = args.iters + args.warmup
    npu_time_total = 0.0
    npu_time_min = float("inf")
    npu_time_max = 0.0
    errors = 0

    # Main run loop
    for iter in range(num_iter):
        if args.verbosity >= 1:
            print(f"\n{'='*60}")
            print(f"Iteration {iter + 1}/{num_iter}")
            print(f"{'='*60}")

        # Run kernel
        if args.verbosity >= 1:
            print("Running Kernel...")

        opcode = 3
        start = time.perf_counter()
        run = kernel(
            opcode,
            bo_instr,
            len(instr_v),
            bo_in1,
            bo_in2,
            bo_out,
            bo_ctrlpkts,
            bo_trace,
        )
        run.wait()
        stop = time.perf_counter()

        # Sync device to host
        bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        if args.trace_sz > 0:
            bo_trace.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        # Skip warmup iterations for timing
        if iter < args.warmup:
            if args.verbosity >= 1:
                print("(Warmup iteration - not counted)")
            continue

        # Read output buffer
        # pyxrt read() returns numpy array directly
        bufOut_bytes = bo_out.read(OUT_SIZE, 0)
        bufOut = np.frombuffer(bufOut_bytes, dtype=DATATYPE_OUT)

        # Verify results
        if args.verify:
            if args.verbosity >= 1:
                print("Verifying results...")

            vstart = time.time()
            errors += verify_vector_scalar_mul(
                bufIn1, bufIn2, bufOut, IN1_VOLUME, args.verbosity
            )
            vstop = time.time()

            if args.verbosity >= 1:
                print(f"Verify time: {vstop - vstart:.3f} secs")
        else:
            if args.verbosity >= 1:
                print("WARNING: Results not verified.")

        # Write trace on first non-warmup iteration
        if args.trace_sz > 0 and iter == args.warmup:
            # Read trace buffer (buffer 7)
            trace_data_bytes = bo_trace.read(args.trace_sz, 0)
            # Convert to uint32 array for proper formatting
            trace_buffer = np.frombuffer(trace_data_bytes, dtype=np.uint32)

            if args.verbosity >= 1:
                print(f"Trace buffer shape: {trace_buffer.shape}")
                print(f"Trace buffer dtype: {trace_buffer.dtype}")

            write_out_trace(trace_buffer, args.trace_file)

        # Accumulate timing
        npu_time = (stop - start) * 1e6  # Convert to microseconds
        npu_time_total += npu_time
        npu_time_min = min(npu_time, npu_time_min)
        npu_time_max = max(npu_time, npu_time_max)

        if args.verbosity >= 1:
            print(f"NPU time: {npu_time:.2f} μs")

    # Print summary
    n_iterations = args.iters
    if n_iterations > 0:
        npu_time_avg = npu_time_total / n_iterations
    else:
        npu_time_avg = 0

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Avg NPU time: {npu_time_avg:.2f} μs")
    print(f"Min NPU time: {npu_time_min:.2f} μs")
    print(f"Max NPU time: {npu_time_max:.2f} μs")
    print(f"Iterations:   {n_iterations}")
    print("=" * 80)

    if args.verify:
        if errors == 0:
            print("\n✓ TEST PASSED - All results verified correctly!")
            return 0
        else:
            print(f"\n✗ TEST FAILED - {errors} errors detected!")
            return 1
    else:
        print("\n? TEST SKIPPED - Verification disabled")
        return 0


if __name__ == "__main__":
    sys.exit(main())
