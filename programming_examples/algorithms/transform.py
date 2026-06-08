# transform.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Tutorial: tile-by-tile elementwise transform on the NPU.

Applies ``lambda a: a + 1`` to each ``tile_size``-element tile of a 1-D
int32 tensor.  The design body delegates to
:func:`aie.iron.algorithms.transform_typed`, which handles the
ObjectFifo / Worker / Runtime plumbing for any single-input element-wise
lambda.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out
from aie.iron.algorithms import transform_typed
from aie.utils.verify import assert_pass


@iron.jit
def transform(
    input: In,
    output: Out,
    *,
    num_elements: Compile[int],
    dtype: Compile[type],
    tile_size: Compile[int] = 16,
):
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    return transform_typed(lambda a: a + 1, tensor_ty, tile_size=tile_size)


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
    input = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)

    transform(input, output, num_elements=int(input.shape[0]), dtype=dtype)

    if args.verbose:
        print(f"{'input':>6} + 1 = {'output':>6}")
        print("-" * 24)
        n = args.num_elements
        for idx, (a, b) in enumerate(zip(input[:n], output[:n])):
            print(f"{idx:2}: {a:6} + 1 = {b:6}")

    assert_pass(input.numpy() + 1, output.numpy(), fail_msg="transform output mismatch")


if __name__ == "__main__":
    main()
