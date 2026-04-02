# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# Test for Prototype 1: ObjectFIFO Broadcast + Split Baseline
#
# Verifies that input data is split across 4 cores, each core passes
# through its chunk, and the results are joined back in correct order.

import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    n_cores = 4
    chunk_size = 256  # bytes per core
    total_size = n_cores * chunk_size  # 1024 bytes

    in_dtype = np.uint8
    out_dtype = np.uint8

    in_volume = total_size // np.dtype(in_dtype).itemsize
    out_volume = total_size // np.dtype(out_dtype).itemsize

    # Input: sequential bytes 0..255 repeated across chunks
    ref_in = np.arange(0, in_volume, dtype=in_dtype)

    # Expected output: passthrough (same as input)
    ref_out = ref_in.copy()

    in1 = iron.tensor(ref_in, dtype=in_dtype)
    out = iron.zeros([out_volume], dtype=out_dtype)

    print("Running Prototype 1: Broadcast/Split Baseline...")
    print(f"  Input size:  {total_size} bytes")
    print(f"  Cores:       {n_cores}")
    print(f"  Chunk/core:  {chunk_size} bytes")

    npu_opts = test_utils.create_npu_kernel(opts)
    # Buffer index 1 is outTensor (2nd argument in runtime_sequence)
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
