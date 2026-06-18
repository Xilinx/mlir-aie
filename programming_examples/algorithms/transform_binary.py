# transform_binary.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Tutorial: tile-by-tile two-input elementwise transform on the NPU.

Applies ``lambda a, b: a + b`` to each ``tile_size``-element tile of two
1-D int32 tensors.  The design body delegates to
:func:`aie.iron.algorithms.transform_binary_typed`.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, Out
from aie.iron.algorithms import transform_binary_typed
from aie.utils.verify import assert_pass


@iron.jit
def transform_binary(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: CompileTime[int],
    dtype: CompileTime[type],
    tile_size: CompileTime[int] = 16,
):
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    return transform_binary_typed(lambda a, b: a + b, tensor_ty, tile_size=tile_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print every output element"
    )
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=1024,
        help="number of int32 elements per input tensor (default: %(default)s)",
    )
    args = parser.parse_args()

    dtype = np.int32
    input0 = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    transform_binary(
        input0, input1, output, num_elements=int(input0.shape[0]), dtype=dtype
    )

    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        n = args.num_elements
        for idx, (a, b, c) in enumerate(zip(input0[:n], input1[:n], output[:n])):
            print(f"{idx:2}: {a:4} + {b:4} = {c:4}")

    assert_pass(
        input0.numpy() + input1.numpy(),
        output.numpy(),
        fail_msg="transform_binary output mismatch",
    )


if __name__ == "__main__":
    main()
