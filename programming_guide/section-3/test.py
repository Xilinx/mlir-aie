# test.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import numpy as np
import pyxrt as xrt
import sys
import time

# ------------------------------------------------------
# Configure this to match your design's buffer size
# ------------------------------------------------------
INOUT0_VOLUME = 64  # Input only, 64x uint32_t in this example
INOUT1_VOLUME = 64  # Not used in this example
INOUT2_VOLUME = 64  # Output only, 64x uint32_t in this example

INOUT0_DATATYPE = np.uint32
INOUT1_DATATYPE = np.uint32
INOUT2_DATATYPE = np.uint32

INOUT0_SIZE = INOUT0_VOLUME * INOUT0_DATATYPE().itemsize
INOUT1_SIZE = INOUT1_VOLUME * INOUT1_DATATYPE().itemsize
INOUT2_SIZE = INOUT2_VOLUME * INOUT2_DATATYPE().itemsize


# options
def parse_args(args):
    p = argparse.ArgumentParser()
    p.add_argument(
        "-x", "--xclbin", required=True, dest="xclbin", help="the input xclbin path"
    )
    p.add_argument(
        "-k",
        "--kernel",
        required=True,
        dest="kernel",
        default="MLIR_AIE",
        help="the kernel name in the XCLBIN (for instance MLIR_AIE)",
    )
    p.add_argument(
        "-v", "--verbosity", default=0, type=int, help="the verbosity of the output"
    )
    p.add_argument(
        "-i",
        "--instr",
        dest="instr",
        default="instr.txt",
        help="path of file containing userspace instructions sent to the NPU",
    )
    p.add_argument(
        "--verify",
        dest="verify",
        default=True,
        help="whether to verify the AIE computed output",
    )
    p.add_argument(
        "--iters",
        dest="iters",
        default=1,
        type=int,
        help="number of benchmark iterations",
    )
    p.add_argument(
        "--warmup",
        dest="warmup_iters",
        default=0,
        type=int,
        help="number of warmup iterations",
    )
    p.add_argument(
        "-t",
        "--trace_sz",
        dest="trace_size",
        default=0,
        type=int,
        help="trace size in bytes",
    )
    p.add_argument(
        "--trace_file",
        dest="trace_file",
        default="trace.txt",
        help="where to store trace output",
    )
    return p.parse_args(args)


def main(opts):

    # Load instruction sequence
    with open(opts.instr, "r") as f:
        instr_text = f.read().split("\n")
        instr_text = [l for l in instr_text if l != ""]
        instr_v = np.array([int(i, 16) for i in instr_text], dtype=np.uint32)

    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------

    # Get a device handle
    device = xrt.device(0)

    # Load the xclbin
    xclbin = xrt.xclbin(opts.xclbin)

    # Load the kernel
    kernels = xclbin.get_kernels()
    try:
        xkernel = [k for k in kernels if opts.kernel in k.get_name()][0]
    except:
        print(f"Kernel '{opts.kernel}' not found in '{opts.xclbin}'")
        exit(-1)

    # Register xclbin
    device.register_xclbin(xclbin)

    # Get a hardware context
    context = xrt.hw_context(device, xclbin.get_uuid())

    # get a kernel handle
    kernel = xrt.kernel(context, xkernel.get_name())

    # ------------------------------------------------------
    # Initialize input/ output buffer sizes and sync them
    # ------------------------------------------------------
    bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(0))
    bo_inout0 = xrt.bo(device, INOUT0_SIZE, xrt.bo.host_only, kernel.group_id(2))
    bo_inout1 = xrt.bo(device, INOUT1_SIZE, xrt.bo.host_only, kernel.group_id(3))
    bo_inout2 = xrt.bo(device, INOUT2_SIZE, xrt.bo.host_only, kernel.group_id(4))

    # Initialize instruction buffer
    bo_instr.write(instr_v, 0)

    # Initialize data buffers
    inout0 = np.arange(1, INOUT0_VOLUME + 1, dtype=INOUT0_DATATYPE)
    inout1 = np.zeros(INOUT1_VOLUME, dtype=INOUT1_DATATYPE)
    inout2 = np.zeros(INOUT2_VOLUME, dtype=INOUT2_DATATYPE)
    bo_inout0.write(inout0, 0)
    bo_inout1.write(inout1, 0)
    bo_inout2.write(inout2, 0)

    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout0.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout1.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    num_iter = opts.iters + opts.warmup_iters
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    errors = 0

    for i in range(num_iter):
        if opts.verbosity >= 1:
            print("Running Kernel.")
        start = time.time_ns()
        h = kernel(bo_instr, len(instr_v), bo_inout0, bo_inout1, bo_inout2)
        h.wait()
        stop = time.time_ns()
        bo_inout2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        if i < opts.warmup_iters:
            continue

        out_size = INOUT2_SIZE + opts.trace_size
        output_buffer = bo_inout2.read(out_size, 0).view(INOUT2_DATATYPE)
        if opts.verify:
            if opts.verbosity >= 1:
                print("Verifying results ...")
            ref = np.arange(2, INOUT0_VOLUME + 2, dtype=INOUT0_DATATYPE)
            e = np.equal(output_buffer, ref)
            errors = errors + np.size(e) - np.count_nonzero(e)

        if opts.trace_size > 0:
            print("Do something with trace!")

        npu_time = stop - start
        npu_time_total = npu_time_total + npu_time
        npu_time_min = min(npu_time_min, npu_time)
        npu_time_max = max(npu_time_max, npu_time)

    # ------------------------------------------------------
    # Print verification and timing results
    # ------------------------------------------------------

    # TODO - Mac count to guide gflops

    print("\nAvg NPU time: {}us.".format(int((npu_time_total / opts.iters) / 1000)))
    print("\nMin NPU time: {}us.".format(int((npu_time_min / opts.iters) / 1000)))
    print("\nMax NPU time: {}us.".format(int((npu_time_max / opts.iters) / 1000)))

    if not errors:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    opts = parse_args(sys.argv[1:])
    main(opts)
