# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DEFAULT_NPU_RUNTIME


def main(opts):
    print("Running...\n")

    data_size = int(opts.size)
    dtype = np.uint8

    input_data = np.arange(1, data_size + 1, dtype=dtype)
    in1 = iron.tensor(input_data, dtype=dtype)
    out = iron.zeros(data_size, dtype=dtype)

    npu_opts = test_utils.create_npu_kernel(opts)
    res = DEFAULT_NPU_RUNTIME.run_test(
        [in1, out],
        [(1, input_data)],
        npu_opts.npu_kernel,
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-s", "--size", required=True, dest="size", help="Passthrough kernel size"
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
