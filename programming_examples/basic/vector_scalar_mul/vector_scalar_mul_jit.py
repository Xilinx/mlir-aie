# vector_scalar_mul/vector_scalar_mul_jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
import sys
import numpy as np
import os
import argparse
import time

import aie.iron as iron
from aie.iron.algorithms import transform


def vector_scalar_mul(input, factor, output):

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

    vectorized = True

    # Define tensor types
    tile_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # Create a handle to an externally-defined kernel
    func_type = "vector" if vectorized else "scalar"

    kernels_path = os.path.join(os.path.dirname(__file__), "../../../aie_kernels/aie2")

    scale_kernel = iron.ExternalFunction(
        f"vector_scalar_mul_{func_type}",
        source_file=os.path.join(kernels_path, "scale.cc"),
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
        include_dirs=[kernels_path],
        compile_flags=["-DBIT_WIDTH=16"],
    )

    # Pass scale kernel to the transform algorithm
    # Note tile_size is passed in here because it is required by the scale kernel
    # iron.jit compiles and runs the program
    iron.jit(is_placed=False)(transform)(scale_kernel, input, output, factor, tile_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=2048,
        help="Number of elements (default: 2048)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=20,
        help="Number of measurement iterations (default: 20)",
    )

    args = parser.parse_args()
    num_elements = args.num_elements
    n_warmup_iterations = args.warmup
    n_iterations = args.iters

    # Initialize timing variables
    npu_time_total = 0.0
    npu_time_min = float("inf")
    npu_time_max = 0.0

    # Tensor setup
    input = iron.randint(0, 100, (num_elements,), dtype=np.int16, device="npu")
    factor = iron.tensor([3], dtype=np.int32, device="npu")
    output = iron.tensor((num_elements,), dtype=np.int16, device="npu")

    # Main run loop with warmup and measurement iterations
    total_iterations = n_warmup_iterations + n_iterations
    for iter_num in range(total_iterations):
        # Launch the kernel and measure execution time
        start_time = time.perf_counter()
        vector_scalar_mul(input, factor, output)
        end_time = time.perf_counter()

        # Calculate execution time in microseconds
        execution_time_us = (end_time - start_time) * 1_000_000

        # Skip warmup iterations for timing statistics
        if iter_num >= n_warmup_iterations:
            npu_time_total += execution_time_us
            npu_time_min = min(npu_time_min, execution_time_us)
            npu_time_max = max(npu_time_max, execution_time_us)

    # Check the correctness of the result
    computed = output.numpy()
    expected = input.numpy() * factor.numpy()[0]
    errors = np.sum(computed != expected)

    if errors == 0:
        # Print timing results
        if n_iterations > 1:
            avg_time = npu_time_total / n_iterations
            print(f"\nAvg NPU time: {avg_time:.1f}us.")
            print(f"Min NPU time: {npu_time_min:.1f}us.")
            print(f"Max NPU time: {npu_time_max:.1f}us.")
        else:
            print(f"\nNPU time: {npu_time_total:.1f}us.")
        print("PASS!")
        sys.exit(0)
    else:
        print(f"FAIL!: Expected {expected} but got {computed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
