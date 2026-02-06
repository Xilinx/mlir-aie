# transform_parallel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
import argparse
import sys
import numpy as np
import aie.iron as iron

from aie.iron.algorithms import transform_parallel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=1024,
        help="Number of elements (default: 1024)",
    )
    args = parser.parse_args()

    dtype = np.int32

    # Construct two input random tensors and an output zeroed tensor
    # The three tensor are in memory accessible to the NPU
    input = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)

    # JIT compile the algorithm
    iron.jit(is_placed=False)(transform_parallel)(lambda a: a + 1, input, output)

    # Check the correctness of the result
    e = np.equal(input.numpy() + 1, output.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # Optionally, print the results
    if args.verbose:
        print(f"Input shape: {input.shape}")
        print(f"Input dtype: {input.dtype}")

        print(f"{'input':>4} + {'output':>4}")
        print("-" * 34)
        count = input.numel()
        # print the first 10 elements
        for idx, (a, b) in enumerate(zip(input[:10], output[:10])):
            print(f"{idx:2}: {a:4} + {b:4}")

    # If the result is correct, exit with a success code.
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        sys.exit(-1)


if __name__ == "__main__":
    main()
