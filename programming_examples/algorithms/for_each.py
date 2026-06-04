# for_each.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Tutorial: in-place tile-by-tile transform on the NPU.

Applies ``lambda a: a + 1`` to each ``tile_size``-element tile of a
single 1-D int32 tensor in place via :func:`aie.iron.algorithms.for_each`.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron.algorithms import for_each
from aie.utils.verify import assert_pass


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
        help="number of int32 elements in the input tensor (default: %(default)s)",
    )
    args = parser.parse_args()

    dtype = np.int32
    tensor = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    initial = tensor.numpy().copy()

    iron.jit(for_each)(
        tensor,
        func=lambda a: a + 1,
        N=int(tensor.shape[0]),
        dtype=tensor.dtype,
        tile_size=16,
    )

    if args.verbose:
        print(f"{'input':>6} + 1 = {'output':>6}")
        print("-" * 22)
        n = args.num_elements
        for idx, (a, b) in enumerate(zip(initial[:n], tensor[:n])):
            print(f"{idx:2}: {a:6} + 1 = {b:6}")

    assert_pass(initial + 1, tensor.numpy(), fail_msg="for_each output mismatch")


if __name__ == "__main__":
    main()
