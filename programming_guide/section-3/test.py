# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
from pathlib import Path
import sys

import aie.iron as iron
import aie.utils
import aie.utils.test as test_utils
from aie.utils.npukernel import NPUKernel


def main(opts):
    # ------------------------------------------------------------
    # Configure this to match your design's buffer size and type
    # ------------------------------------------------------------

    # Initialize data buffers and reference for verification
    ref_buffer = np.arange(1, 64 + 1, dtype=np.int32)
    in_buffer = iron.tensor(ref_buffer, dtype=np.int32)
    scale_factor = 3
    in_factor = iron.tensor([scale_factor], dtype=np.int32)
    out = iron.zeros(64, dtype=np.int32)
    ref_buffer = ref_buffer * scale_factor

    # ----------------------------------------------------
    # Prepare buffers and load compiled artifacts onto the device
    # ----------------------------------------------------
    npu_kernel = NPUKernel(opts.xclbin, opts.instr)
    kernel_handle = aie.utils.DEFAULT_NPU_RUNTIME.load(npu_kernel)

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
    npu_time = aie.utils.DEFAULT_NPU_RUNTIME.run(
        kernel_handle, [in_buffer, in_factor, out]
    )

    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")
        e = np.equal(out, ref_buffer)
        errors = errors + np.size(e) - np.count_nonzero(e)

    # ------------------------------------------------------
    # Print verification and timing results
    # ------------------------------------------------------

    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
