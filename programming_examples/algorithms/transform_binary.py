# transform_binary.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Tutorial: tile-by-tile two-input elementwise transform on the NPU.

Applies ``lambda a, b: a + b`` to each ``tile_size``-element tile of two
1-D int32 tensors via :func:`aie.iron.algorithms.transform_binary`.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron.algorithms import transform_binary
from aie.utils.verify import assert_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--num-elements", type=int, default=1024)
    args = parser.parse_args()

    dtype = np.int32
    input0 = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    iron.jit(transform_binary)(
        input0,
        input1,
        output,
        func=lambda a, b: a + b,
        N=int(input0.shape[0]),
        dtype=input0.dtype,
        tile_size=16,
    )

    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        for idx, (a, b, c) in enumerate(zip(input0[:10], input1[:10], output[:10])):
            print(f"{idx:2}: {a:4} + {b:4} = {c:4}")

    assert_pass(
        input0.numpy() + input1.numpy(),
        output.numpy(),
        fail_msg="transform_binary output mismatch",
    )


if __name__ == "__main__":
    main()
