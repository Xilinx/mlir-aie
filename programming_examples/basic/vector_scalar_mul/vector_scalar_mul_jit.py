# vector_scalar_mul/vector_scalar_mul_jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import os

import aie.iron as iron
from aie.iron.algorithms import transform


def vector_scalar_mul(input, factor, output, dummy_input0, trace):

    in1_dtype = input.dtype
    tensor_size = input.numel()
    num_sub_vectors = 4
    tile_size = tensor_size // num_sub_vectors

    if input.dtype != np.int16:
        raise ValueError("Input must be of type int16")

    if factor.dtype != np.int32:
        raise ValueError("Factor must be of type int32")

    if output.dtype != np.int16:
        raise ValueError("Output must be of type int16")

    if input.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({input.shape} != {output.shape})."
        )

    if factor.numel() != 1:
        raise ValueError(f"Factor must be a scalar, but has shape {factor.shape}.")

    enable_trace = 1 if trace.numel() > 0 else 0

    vectorized = True

    # Define tensor types
    tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # Create a handle to an externally-defined kernel
    func_type = "vector" if vectorized else "scalar"

    kernels_path = os.path.join(os.path.dirname(__file__), "../../../aie_kernels/aie2")

    scale = iron.ExternalFunction(
        f"vector_scalar_mul_{func_type}",
        source_file=os.path.join(kernels_path, "scale.cc"),
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
        include_dirs=[kernels_path],
    )

    # Pass scale kernel to the transform algorithm
    # 'factor' is passed as an extra argument; tile_size is auto-provided
    iron.jit(is_placed=False)(transform)(input, output, scale, factor)


def main():

    num_elements = 1024

    # Tensor setup
    input = iron.randint(0, 100, (num_elements,), dtype=np.int16, device="npu")
    factor = iron.tensor([3], dtype=np.int32, device="npu")
    dummy_input = iron.zeros_like(input)
    output = iron.zeros_like(input)
    trace = iron.zeros(128, dtype=np.uint32)

    # Function call
    vector_scalar_mul(input, factor, output, dummy_input, trace)

    with open("trace.txt", "w") as f:
        for val in trace:
            f.write(f"{val:08x}\n")


if __name__ == "__main__":
    main()
