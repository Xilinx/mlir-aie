# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import argparse
from aie.utils.hostruntime.argparse import add_runtime_args
from aie.utils.test import create_npu_kernel
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    print("Running...\n")

    data_size = int(opts.size)
    dtype = np.uint8

    in1 = iron.arange(1, data_size + 1, dtype=dtype)
    out = iron.zeros(data_size, dtype=dtype)
    input_data = in1.numpy()

    npu_opts = create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, out],
        {1: input_data},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_runtime_args(p, with_io_sizes=True)
    p.add_argument(
        "-s", "--size", required=True, dest="size", help="Passthrough kernel size"
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
