# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Test for Prototype 3: Time-Multiplexed Multi-Phase Pipeline

import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    n_cores = 4
    chunk_size = 256
    n_iterations = 4
    batch_size = n_cores * chunk_size
    total_size = n_iterations * batch_size

    in_dtype = np.uint8
    out_dtype = np.uint8

    in_volume = total_size // np.dtype(in_dtype).itemsize
    out_volume = total_size // np.dtype(out_dtype).itemsize

    ref_in = np.arange(0, in_volume, dtype=in_dtype)
    ref_out = ref_in.copy()

    in1 = iron.tensor(ref_in, dtype=in_dtype)
    out = iron.zeros([out_volume], dtype=out_dtype)

    print("Running Prototype 3: Time-Multiplexed Multi-Phase...")
    print(f"  Total input:   {total_size} bytes")
    print(f"  Iterations:    {n_iterations}")
    print(f"  Batch/iter:    {batch_size} bytes")
    print(f"  Cores:         {n_cores}")
    print(f"  Chunk/core:    {chunk_size} bytes")
    print(f"  ShimDMA:       1 MM2S + 1 S2MM (reused across {n_iterations} iterations)")

    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, out],
        {1: ref_out},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    else:
        print("\nFAIL!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
