# vector_reduce_min/vector_reduce_min.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import os
import argparse

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron import ExternalFunction


@iron.jit(is_placed=False)
def my_reduce_min(input_tensor, output_tensor):

    num_elements = input_tensor.numel()
    assert output_tensor.numel() == 1, "Output tensor must be a scalar"

    # Define tensor types
    in_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    out_ty = np.ndarray[(1,), np.dtype[output_tensor.dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(in_ty, name="in")
    of_out = ObjectFifo(out_ty, name="out")

    # AIE Core Function declarations
    root_dir = os.path.abspath(os.path.join(__file__, "../../../.."))
    kernel_dir = os.path.join(root_dir, "aie_kernels/aie2")
    source_file = os.path.join(kernel_dir, "reduce_min.cc")
    reduce_min_vector = ExternalFunction(
        "reduce_min_vector",
        source_file=source_file,
        arg_types=[in_ty, out_ty, np.int32],
        include_dirs=[kernel_dir],
    )

    # Define a task
    def core_body(of_in, of_out, reduce_min_vector):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        reduce_min_vector(elem_in, elem_out, num_elements)
        of_in.release(1)
        of_out.release(1)

    # Define a worker to run the task on a core
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), reduce_min_vector])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=1024,
        help="Number of elements (default: 1024)",
    )

    args = parser.parse_args()
    num_elements = args.num_elements
    data_type = np.int32

    # Construct input and output tensors that are accessible to the NPU
    input_tensor = iron.randint(10, 100, (num_elements,), dtype=data_type, device="npu")
    output_tensor = iron.tensor((1,), dtype=data_type, device="npu")

    # Launch the kernel
    my_reduce_min(input_tensor, output_tensor)

    # Check the correctness of the result
    expected = input_tensor.numpy().min()
    computed = output_tensor.numpy()[0]

    if expected == computed:
        print("PASS!")
        sys.exit(0)
    else:
        print(f"FAIL!: Expected {expected} but got {computed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
