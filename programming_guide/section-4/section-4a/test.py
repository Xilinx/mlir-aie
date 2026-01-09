# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys
from pathlib import Path

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
    kernel_handle = aie.utils.DefaultNPURuntime.load(npu_kernel)

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
        result = aie.utils.DefaultNPURuntime.run(
            kernel_handle, [in_buffer, in_factor, out]
        )
        npu_time = result.npu_time

        # Warmup iterations do not count towards average runtime.
        if i < opts.warmup_iters:
            continue

        # Copy output results and verify they are correct
        if opts.verify:
            if opts.verbosity >= 1:
                print("Verifying results ...")
            e = np.equal(out, ref_buffer)
            errors = errors + np.size(e) - np.count_nonzero(e)

        npu_time_total = npu_time_total + npu_time
        npu_time_min = min(npu_time_min, npu_time)
        npu_time_max = max(npu_time_max, npu_time)

    # ------------------------------------------------------
    # Print verification and timing results
    # ------------------------------------------------------

    # TODO - Mac count to guide gflops

    print(
        "\nNumber of iterations:",
        str(opts.iters),
        "(warmup iterations:",
        str(opts.warmup_iters),
        ")",
    )

    print("\nAvg NPU time: {}us.".format(int((npu_time_total / opts.iters) / 1000)))
    print("\nMin NPU time: {}us.".format(int((npu_time_min) / 1000)))
    print("\nMax NPU time: {}us.".format(int((npu_time_max) / 1000)))

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
