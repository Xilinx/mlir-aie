# vector_vector_add/vector_vector_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise vector + vector add — Iron API design with ``@iron.jit``.

The design body delegates to ``aie.iron.algorithms.transform_binary_typed``,
which handles the ObjectFifo / Worker / Runtime plumbing for any binary
element-wise lambda.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out
from aie.iron.algorithms import transform_binary_typed
from aie.utils.verify import assert_pass


@iron.jit
def vector_vector_add(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: Compile[int],
    dtype: Compile[type],
    tile_size: Compile[int] = 16,
):
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    return transform_binary_typed(lambda a, b: a + b, tensor_ty, tile_size=tile_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--num-elements", type=int, default=32)
    args = parser.parse_args()

    input0 = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    input1 = iron.randint(0, 100, (args.num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)

    if input0.shape != input1.shape or input0.shape != output.shape:
        raise ValueError("All three tensors must share the same shape.")
    if input0.dtype != input1.dtype or input0.dtype != output.dtype:
        raise ValueError("All three tensors must share the same dtype.")
    if len(input0.shape) != 1:
        raise ValueError("Function only supports vectors.")

    vector_vector_add(
        input0,
        input1,
        output,
        num_elements=int(np.size(input0)),
        dtype=input0.dtype,
    )

    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        for idx, (a, b, c) in enumerate(zip(input0[:10], input1[:10], output[:10])):
            print(f"{idx:2}: {a:4} + {b:4} = {c:4}")

    assert_pass(
        input0.numpy() + input1.numpy(),
        output.numpy(),
        fail_msg="vector_vector_add output mismatch",
    )


if __name__ == "__main__":
    main()
