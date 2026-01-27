# transform_binary.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys
import numpy as np
import aie.iron as iron
import tempfile
import os


from aie.iron.algorithms import for_each
from aie.iron import CoreFunction


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
    tensor = iron.randint(0, 100, (args.num_elements,), dtype=dtype, device="npu")
    initial_tensor = tensor.numpy().copy()

    # Create external function at tmp file

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_name = tmp_file.name + ".cc"
        print(f"tmp_file_name: {tmp_file_name}")
        with open(tmp_file_name, "w") as f:
            f.write("""extern "C" {
                    void add_one(int* input, int* output, int tile_size) {
                        for (int i = 0; i < tile_size; i++) {
                            output[i] = input[i] + 1;
                        }
                    }
                }"""
            )
        add_one = CoreFunction(
            f"add_one",
            source_file=tmp_file_name,
            arg_types=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
                np.int32,
            ],
        )

    # Create external function
    for_each(tensor, add_one)

    print(f"initial_tensor: {initial_tensor}")
    print(f"output tensor: {tensor}")

    # Check the correctness of the result
    e = np.equal(initial_tensor + 1, tensor.numpy())
    errors = np.size(e) - np.count_nonzero(e)

    # Optionally, print the results
    if args.verbose:
        print(f"{'input0':>4} + {'input1':>4} = {'output':>4}")
        print("-" * 34)
        count = tensor.numel()
        for idx, (a) in enumerate(zip(tensor[:count])):
            print(f"{idx:2}: {a:4}")

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
