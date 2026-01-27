# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    in1_size = int(opts.in1_size)  # in bytes
    out_size = int(opts.out_size)  # in bytes

    # --------------------------------------------------------------------------
    # ----- Edit your data types -----------------------------------------------
    # --------------------------------------------------------------------------

    in1_dtype = np.uint8
    out_dtype = in1_dtype

    # --------------------------------------------------------------------------

    in1_volume = in1_size // np.dtype(in1_dtype).itemsize
    out_volume = out_size // np.dtype(out_dtype).itemsize

    # --------------------------------------------------------------------------
    # ----- Edit your data init and reference data here ------------------------
    # --------------------------------------------------------------------------

    # check buffer sizes
    assert out_size == in1_size

    # Initialize data
    ref = np.arange(0, in1_volume, dtype=in1_dtype)
    in1 = iron.tensor(ref, dtype=in1_dtype)
    out = iron.zeros([out_volume], dtype=out_dtype)

    # --------------------------------------------------------------------------

    print("Running...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, out],
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
