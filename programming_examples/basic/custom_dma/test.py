# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Test for the custom_dma scatter-read example.

Expected output (48 x i32):
  [0..15]  = row 0: [100, 101, ..., 115]
  [16..31] = row 1: [200, 201, ..., 215]
  [32..47] = row 3: [400, 401, ..., 415]
"""

import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    out_dtype = np.int32
    cols = 16
    out_volume = cols * 3  # three row transfers

    row0 = np.arange(100, 100 + cols, dtype=out_dtype)
    row1 = np.arange(200, 200 + cols, dtype=out_dtype)
    row3 = np.arange(400, 400 + cols, dtype=out_dtype)
    ref = np.concatenate([row0, row1, row3])

    out = iron.zeros([out_volume], dtype=out_dtype)
    dummy = iron.zeros([out_volume], dtype=out_dtype)

    print("Running...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [dummy, out],
        {1: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
