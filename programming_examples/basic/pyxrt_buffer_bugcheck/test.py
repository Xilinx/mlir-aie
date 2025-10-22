# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import aie.utils.xrt as xrt_utils
import aie.utils.test as test_utils


def main(opts):
    assert opts.ncores == 1 or opts.ncores == 2
    in1_size = int(opts.in1_size)  # in bytes
    in2_size = int(opts.in2_size)  # in bytes
    out_size = int(opts.out_size)  # in bytes
    assert in1_size + in2_size == out_size

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

    # Initialize data
    in1_data = np.arange(0, in1_volume, dtype=in1_dtype)
    in2_data = np.arange(0, in1_volume, dtype=in1_dtype)
    out_data = np.zeros([out_volume], dtype=out_dtype)

    # Define reference data
    if opts.ncores == 2:
        in2_dtype, in2_data, in2_volume = (in1_dtype, in2_data, in1_volume)
        ref = np.concatenate((in1_data, in2_data))
    else:
        in2_dtype, in2_data, in2_volume = (None, None, None)
        ref = in1_data

    # --------------------------------------------------------------------------

    print(f"Running... {opts}\n")
    res = xrt_utils.setup_and_run_aie(
        in1_dtype,
        in2_dtype,
        out_dtype,
        in1_data,
        in2_data,
        out_data,
        in1_volume,
        in2_volume,
        out_volume,
        ref,
        opts,
    )
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument('--ncores', type=int)
    opts = p.parse_args()

    main(opts)
