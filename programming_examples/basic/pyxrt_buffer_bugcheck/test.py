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
    in1_size = int(opts.in1_size)  # in bytes
    in2_size = int(opts.in2_size)  # in bytes
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
    in1_data = np.arange(0, in1_volume, dtype=in1_dtype)
    out_data = np.zeros([out_volume], dtype=out_dtype)

    # Define reference data
    ref = in1_data

    # --------------------------------------------------------------------------

    print("Running...\n")
    res = xrt_utils.setup_and_run_aie(
        in1_dtype,
        None,
        out_dtype,
        in1_data,
        None,
        out_data,
        in1_volume,
        None,
        out_volume,
        ref,
        opts,
    )
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
