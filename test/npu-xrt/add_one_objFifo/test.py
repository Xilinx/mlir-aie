# ===- test.py -------------------------------------------------*- Python -*-===#
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# ===----------------------------------------------------------------------===#

# This gets launched from run.lit, so disable it with a bogus requires line
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

import argparse
import pyxrt as xrt
import numpy as np
import os
import struct

IN_SIZE = 64
OUT_SIZE = 64


def check_file_exists(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")


def load_instr_binary(filepath):
    with open(filepath, "rb") as f:
        return list(struct.unpack(f"{os.path.getsize(filepath)//4}I", f.read()))


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="PyXRT Test Script")
    parser.add_argument("-x", "--xclbin", required=True, help="The input xclbin path")
    parser.add_argument(
        "-k",
        "--kernel",
        required=True,
        help="The kernel name in the XCLBIN (e.g., PP_PRE_FD)",
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=0, help="The verbosity of the output"
    )
    parser.add_argument(
        "-i",
        "--instr",
        required=True,
        help="Path of file containing userspace instructions to be sent to the LX6",
    )
    args = parser.parse_args()

    # Check if files exist
    check_file_exists(args.xclbin)
    check_file_exists(args.instr)

    # Load instruction binary
    instr_v = load_instr_binary(args.instr)

    if args.verbosity >= 1:
        print(f"Sequence instr count: {len(instr_v)}")

    device = xrt.device(0)

    # Load the xclbin
    if args.verbosity >= 1:
        print(f"Loading xclbin: {args.xclbin}")
    xclbin = xrt.xclbin(args.xclbin)

    if args.verbosity >= 1:
        print(f"Kernel opcode: {args.kernel}")

    # Get the kernel from the xclbin
    xkernels = xclbin.get_kernels()
    xkernel = [k for k in xkernels if args.kernel in k.get_name()][0]
    kernel_name = xkernel.get_name()

    if args.verbosity >= 1:
        print(f"Registering xclbin: {args.xclbin}")
    device.register_xclbin(xclbin)

    # Get a hardware context
    if args.verbosity >= 1:
        print("Getting hardware context.")
    context = xrt.hw_context(device, xclbin.get_uuid())

    # Get a kernel handle
    if args.verbosity >= 1:
        print(f"Getting handle to kernel: {kernel_name}")
    kernel = xrt.kernel(context, kernel_name)

    # Create buffer objects
    bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(1))
    bo_inA = xrt.bo(device, IN_SIZE * 4, xrt.bo.host_only, kernel.group_id(3))
    bo_inB = xrt.bo(device, IN_SIZE * 4, xrt.bo.host_only, kernel.group_id(4))
    bo_out = xrt.bo(device, OUT_SIZE * 4, xrt.bo.host_only, kernel.group_id(5))

    if args.verbosity >= 1:
        print("Writing data into buffer objects.")

    # Fill input buffer A
    buf_inA = np.arange(1, IN_SIZE + 1, dtype=np.uint32)
    bo_inA.write(buf_inA, 0)

    # Fill instruction buffer
    buf_instr = np.array(instr_v, dtype=np.uint32)
    bo_instr.write(buf_instr, 0)

    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inA.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    if args.verbosity >= 1:
        print("Running Kernel.")

    # Run the kernel
    opcode = 3
    run = kernel(opcode, bo_instr, len(instr_v), bo_inA, bo_inB, bo_out)
    r = run.wait()

    if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
        print(f"Kernel did not complete. Returned status: {r}")
        return 1

    bo_out.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Read output buffer
    buf_out = np.empty(OUT_SIZE, dtype=np.uint32)
    buf_out = bo_out.read(buf_out.size * buf_out.itemsize, 0).view(dtype=buf_out.dtype)

    # Verify output
    errors = 0
    for i in range(OUT_SIZE):
        ref = i + 42
        if buf_out[i] != ref:
            print(f"Error in output {buf_out[i]} != {ref}")
            errors += 1
        else:
            # print(f"Correct output {buf_out[i]} == {ref}")
            pass

    if errors == 0:
        print("\nPASS!\n")
        return 0
    else:
        print("\nfailed.\n")
        return 1


if __name__ == "__main__":
    main()
