# test.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import pyxrt as xrt
import sys
import time

import aie.utils.test as test_utils

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


def main(opts):

    # Load instruction sequence
    with open(opts.instr, "r") as f:
        instr_text = f.read().split("\n")
        instr_text = [l for l in instr_text if l != ""]
        instr_v = np.array([int(i, 16) for i in instr_text], dtype=np.uint32)

    OUT_SIZE = INOUT2_SIZE

    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    (device, kernel) = test_utils.init_xrt_load_kernel(opts)

    # ------------------------------------------------------
    # Initialize input/ output buffer sizes and sync them
    # ------------------------------------------------------
    bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(0))
    bo_inout0 = xrt.bo(device, INOUT0_SIZE, xrt.bo.host_only, kernel.group_id(2))
    bo_inout1 = xrt.bo(device, INOUT1_SIZE, xrt.bo.host_only, kernel.group_id(3))
    # bo_inout2 = xrt.bo(device, INOUT2_SIZE, xrt.bo.host_only, kernel.group_id(4))
    bo_inout2 = xrt.bo(device, OUT_SIZE, xrt.bo.host_only, kernel.group_id(4))

    # Initialize instruction buffer
    bo_instr.write(instr_v, 0)

    # Initialize data buffers
    inout0 = np.arange(1, INOUT0_VOLUME + 1, dtype=INOUT0_DATATYPE)
    inout1 = np.zeros(INOUT1_VOLUME, dtype=INOUT1_DATATYPE)
    inout2 = np.zeros(INOUT2_VOLUME, dtype=INOUT2_DATATYPE)
    bo_inout0.write(inout0, 0)
    bo_inout1.write(inout1, 0)
    bo_inout2.write(inout2, 0)

    # Sync buffers to update input buffer values
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout0.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout1.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # ------------------------------------------------------
    # Initialize run configs
    # ------------------------------------------------------
    num_iter = opts.iters + opts.warmup_iters
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    errors = 0

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    for i in range(num_iter):
        # Run kernel
        if opts.verbosity >= 1:
            print("Running Kernel.")
        start = time.time_ns()
        h = kernel(bo_instr, len(instr_v), bo_inout0, bo_inout1, bo_inout2)
        h.wait()
        stop = time.time_ns()
        bo_inout2.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

        # Warmup iterations do not count towards average runtime.
        if i < opts.warmup_iters:
            continue

        # Copy output results and verify they are correct
        out_size = INOUT2_SIZE
        output_buffer = bo_inout2.read(out_size, 0).view(INOUT2_DATATYPE)
        if opts.verify:
            if opts.verbosity >= 1:
                print("Verifying results ...")
            ref = np.arange(2, INOUT0_VOLUME + 2, dtype=INOUT0_DATATYPE)
            e = np.equal(output_buffer, ref)
            errors = errors + np.size(e) - np.count_nonzero(e)

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
    opts = test_utils.parse_args(sys.argv[1:])
    main(opts)
