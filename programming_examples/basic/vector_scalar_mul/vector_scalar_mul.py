# vector_scalar_mul/vector_scalar_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys
import os

from aie.iron import CoreFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_
import aie.iron as iron

import aie.utils.trace as trace_utils


def my_vector_scalar_mul(dev, in1_size, in2_size, out_size, int_bit_width, trace_size):

    if int_bit_width == 16:
        in1_dtype = np.int16
        out_dtype = np.int16
    else:  # default is 32-bit
        in1_dtype = np.int32
        out_dtype = np.int32


@iron.jit(is_placed=False)
def vector_scalar_mul(input, factor, output, dummy_input0, trace):

    in1_dtype = input.dtype
    in2_dtype = factor.dtype
    out_dtype = output.dtype

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

    scale = iron.CoreFunction(
        f"vector_scalar_mul_int16_{func_type}",
        source_file=os.path.join(kernels_path, "scale.cc"),
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_factor = ObjectFifo(scalar_ty, name="infactor")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task for a compute tile to run
    def core_body(of_in, of_factor, of_out, scale_fn):
        elem_factor = of_factor.acquire(1)

        # Number of sub-vector "tile" iterations
        for _ in range_(num_sub_vectors):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            scale_fn(elem_in, elem_out, elem_factor, tile_size)
            of_in.release(1)
            of_out.release(1)
        of_factor.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale],
        trace=enable_trace,
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (A, F, C):
        rt.enable_trace(trace.numel() * np.dtype(trace.dtype).itemsize)
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.fill(of_factor.prod(), F)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():
    num_elements = 1024

    input = iron.randint(0, 100, (num_elements,), dtype=np.int16, device="npu")
    factor = iron.tensor([3], dtype=np.int32, device="npu")
    dummy_input = iron.zeros_like(input)
    output = iron.zeros_like(input)
    trace = iron.zeros(128, dtype=np.uint32)

    vector_scalar_mul(input, factor, output, dummy_input, trace)
    with open("trace.txt", "w") as f:
        for val in trace:
            f.write(f"{val:08x}\n")


if __name__ == "__main__":
    main()
