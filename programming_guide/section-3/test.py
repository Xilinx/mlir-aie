# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates

import argparse
import numpy as np
from pathlib import Path
import sys

import aie.iron as iron
import aie.utils
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.npukernel import NPUKernel


def main(opts):
    # ------------------------------------------------------------
    # Configure this to match your design's buffer size and type
    # ------------------------------------------------------------

    # Initialize data buffers and reference for verification
    in_buffer = iron.arange(1, 4096 + 1, dtype=np.int32)
    scale_factor = 3
    in_factor = iron.tensor([scale_factor], dtype=np.int32)
    out = iron.zeros(4096, dtype=np.int32)
    ref_buffer = in_buffer.numpy() * scale_factor

    # ----------------------------------------------------
    # Prepare buffers and load compiled artifacts onto the device
    # ----------------------------------------------------
    npu_kernel = NPUKernel(opts.xclbin, opts.instr)
    kernel_handle = aie.utils.DefaultNPURuntime.load(npu_kernel)

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
    npu_time = aie.utils.DefaultNPURuntime.run(
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
    p = argparse.ArgumentParser()
    add_runtime_args(p)
    opts = p.parse_args(sys.argv[1:])
    main(opts)
