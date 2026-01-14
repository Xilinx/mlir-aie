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
    in2_size = int(opts.in2_size)  # in bytes
    out_size = int(opts.out_size)  # in bytes

    # --------------------------------------------------------------------------
    # ----- Edit your data types -----------------------------------------------
    # --------------------------------------------------------------------------

    in1_dtype = np.int32
    in2_dtype = np.int32
    out_dtype = in1_dtype

    # --------------------------------------------------------------------------

    in1_volume = in1_size // np.dtype(in1_dtype).itemsize
    in2_volume = in2_size // np.dtype(in2_dtype).itemsize
    out_volume = out_size // np.dtype(out_dtype).itemsize

    # --------------------------------------------------------------------------
    # ----- Edit your data init and reference data here ------------------------
    # --------------------------------------------------------------------------

    # check buffer sizes
    assert in2_size == 4
    assert out_size == in1_size

    scale_factor = 3

    # Initialize data
    ref = np.arange(1, in1_volume + 1, dtype=in1_dtype)
    in1 = iron.tensor(ref, dtype=in1_dtype)

    in2 = iron.tensor([scale_factor], dtype=in2_dtype)
    out = iron.zeros([out_volume], dtype=out_dtype)

    # Define reference data
    ref = ref * scale_factor

    # --------------------------------------------------------------------------

    print("Running...\n")
    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, in2, out],
        {2: ref},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if not res:
        print("PASS!")
    else:
        print("Failed.")
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
