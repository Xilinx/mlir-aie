# tiling_exploration/test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np

from aie.helpers.taplib import TensorTiler2D
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DEFAULT_NPU_RUNTIME
import sys


def main(opts):
    print("Running...\n")

    dtype = np.int32
    data_size = opts.tensor_height * opts.tensor_width

    reference_tiler = TensorTiler2D.simple_tiler(
        (opts.tensor_height, opts.tensor_width), (opts.tile_height, opts.tile_width)
    )
    reference_access_order = reference_tiler.access_order()

    out = iron.zeros(data_size, dtype=dtype)

    npu_opts = test_utils.create_npu_kernel(opts)
    res = DEFAULT_NPU_RUNTIME.run_test(
        [out],
        {0: reference_access_order.flatten()},
        npu_opts.npu_kernel,
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    sys.exit(res)


def get_arg_parser():
    p = test_utils.create_default_argparser()
    p.add_argument("--tensor-height", required=True, help="Tensor height", type=int)
    p.add_argument("--tensor-width", required=True, help="Tensor width", type=int)
    p.add_argument("--tile-height", required=True, help="Tile height", type=int)
    p.add_argument("--tile-width", required=True, help="Tile width", type=int)
    return p


if __name__ == "__main__":
    p = get_arg_parser()
    opts = p.parse_args()
    main(opts)
