# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import pyxrt as xrt
import sys
import time

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx
from aie.extras.dialects.ext import memref, arith

import aie.utils.test as test_utils


def main(opts):

    print("Running...\n")

    # Load instruction sequence
    with open(opts.instr, "r") as f:
        instr_text = f.read().split("\n")
        instr_text = [l for l in instr_text if l != ""]
        instr_v = np.array([int(i, 16) for i in instr_text], dtype=np.uint32)

    # ------------------------------------------------------------
    # Configure this to match your design's buffer size and type
    # ------------------------------------------------------------
    INOUT0_VOLUME = int(opts.size)  # Input only, 64x uint32_t in this example
    INOUT1_VOLUME = int(opts.size)  # Output only, 64x uint32_t in this example

    INOUT0_DATATYPE = np.uint8
    INOUT1_DATATYPE = np.uint8

    INOUT0_SIZE = INOUT0_VOLUME * INOUT0_DATATYPE().itemsize
    INOUT1_SIZE = INOUT1_VOLUME * INOUT1_DATATYPE().itemsize

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

    # Initialize instruction buffer
    bo_instr.write(instr_v, 0)

    # Initialize data buffers
    inout0 = np.arange(1, INOUT0_VOLUME + 1, dtype=INOUT0_DATATYPE)
    inout1 = np.zeros(INOUT1_VOLUME, dtype=INOUT1_DATATYPE)
    bo_inout0.write(inout0, 0)
    bo_inout1.write(inout1, 0)

    # Sync buffers to update input buffer values
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout0.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_inout1.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # ------------------------------------------------------
    # Initialize run configs
    # ------------------------------------------------------
    errors = 0

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------

    # Run kernel
    if opts.verbosity >= 1:
        print("Running Kernel.")
    h = kernel(bo_instr, len(instr_v), bo_inout0, bo_inout1)
    h.wait()
    bo_inout1.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Copy output results and verify they are correct
    out_size = INOUT1_SIZE + int(opts.trace_size)
    output_buffer = bo_inout1.read(INOUT1_SIZE, 0).view(INOUT1_DATATYPE)
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")
        ref = np.arange(1, INOUT0_VOLUME + 1, dtype=INOUT0_DATATYPE)
        e = np.equal(output_buffer, ref)
        errors = errors + np.size(e) - np.count_nonzero(e)

    # ------------------------------------------------------
    # Print verification and timing results
    # ------------------------------------------------------

    if not errors:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-s", "--size", required=True, dest="size", help="Passthrough kernel size"
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
