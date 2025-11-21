# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from ml_dtypes import bfloat16
import numpy as np
import sys
import os

import aie.iron as iron
from aie.iron import ExternalFunction
from aie.iron import ObjectFifo, Program, Runtime, Worker, Buffer
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape
from aie.helpers.dialects.ext.scf import if_, else_
from aie.utils.config import cxx_header_path


# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - is_placed (bool): Whether the kernel is using explicit or deferred placement API. Defaults to True.
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.
@iron.jit(is_placed=False)
def vector_reduce_max(input0, output):
    element_type = output.dtype

    in_tensor_size = input0.shape[0]  # Input tensor size
    out_tensor_size = output.shape[0]  # Output tensor size

    n_cores = 4
    N = 2048
    elems_per_core = N // n_cores

    num_iter = in_tensor_size // N

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(in_tensor_size,), np.dtype[element_type]]
    mem_ty = np.ndarray[(N,), np.dtype[element_type]]
    op_ty = np.ndarray[(elems_per_core,), np.dtype[element_type]]
    out_ty = np.ndarray[(out_tensor_size,), np.dtype[element_type]]

    # Input A and Output C
    of_in = ObjectFifo(mem_ty, name="of_in")

    in_fifos = []
    out_fifos = []

    if n_cores > 1:
        of_a_offsets = [
            (np.prod(np_ndarray_type_get_shape(mem_ty)) // n_cores) * i
            for i in range(n_cores)
        ]
    else:
        of_a_offsets = [0]

    in_fifos = of_in.cons().split(
        of_a_offsets,
        obj_types=[op_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )

    min_val = np.array([bfloat16(float("-inf"))], dtype=element_type)
    nextC_buffers = []
    tmp_buffers = []
    for i in range(n_cores):
        out_fifos.append(ObjectFifo(out_ty, name=f"memC{i}"))
        nextC_buffers.append(
            Buffer(
                type=np.ndarray[(out_tensor_size,), np.dtype[element_type]],
                initial_value=min_val,
            )
        )
        tmp_buffers.append(
            Buffer(
                type=np.ndarray[(out_tensor_size,), np.dtype[element_type]],
                initial_value=min_val,
            )
        )
    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    reduce_max_vector = ExternalFunction(
        f"reduce_max_vector_bfloat16",
        source_file=os.path.join(os.path.dirname(__file__), "reduce_max_vector.cc"),
        arg_types=[op_ty, out_ty, np.int32],
        include_dirs=[cxx_header_path()],
    )

    def start_core_body(of_in, of_out, reduce_fn, nextC_buffer, tmp_buffer):
        elem_out = of_out.acquire(1)
        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_fn(elem_in, tmp_buffer, elems_per_core)
            with if_(nextC_buffer[0] < tmp_buffer[0]) as if_op:
                nextC_buffer[0] = tmp_buffer[0]
            of_in.release(1)
        elem_out[0] = nextC_buffer[0]
        of_out.release(1)

    def core_body(of_in, of_out, in0, reduce_fn, nextC_buffer, tmp_buffer):
        for _ in range_(num_iter):
            elem_in = of_in.acquire(1)
            reduce_fn(elem_in, tmp_buffer, elems_per_core)
            with if_(nextC_buffer[0] < tmp_buffer[0]) as if_op:
                nextC_buffer[0] = tmp_buffer[0]
            of_in.release(1)

        elem_out = of_out.acquire(1)
        elem_in1 = in0.acquire(1)
        with if_(elem_in1[0] > nextC_buffer[0]) as if_op:
            elem_out[0] = elem_in1[0]
        with else_(if_op):
            elem_out[0] = nextC_buffer[0]
        in0.release(1)
        of_out.release(1)

    workers = []
    for i in range(n_cores):
        if i != n_cores - 1:
            workers.append(
                Worker(
                    core_body,
                    fn_args=[
                        in_fifos[i].cons(),
                        out_fifos[i].prod(),
                        out_fifos[i + 1].cons(),
                        reduce_max_vector,
                        nextC_buffers[i],
                        tmp_buffers[i],
                    ],
                    trace=None,
                )
            )
        else:
            workers.append(
                Worker(
                    start_core_body,
                    fn_args=[
                        in_fifos[i].cons(),
                        out_fifos[i].prod(),
                        reduce_max_vector,
                        nextC_buffers[i],
                        tmp_buffers[i],
                    ],
                    trace=None,
                )
            )

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(*workers)
        rt.fill(of_in.prod(), a_in)
        rt.drain(out_fifos[0].cons(), c_out, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())


def main():
    # Define tensor shapes and data types
    in_size = 524288
    out_size = 4
    element_type = bfloat16

    assert out_size == 4, "Output buffer must be size 4 (4 bytes = 1 integer)."

    in_tensor_size = in_size // element_type(0).nbytes
    out_tensor_size = out_size // element_type(0).nbytes

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(in_tensor_size, dtype=element_type, device="npu")
    output = iron.arange(out_tensor_size, dtype=element_type, device="npu")

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    vector_reduce_max(input0, output)

    # Check the correctness of the result and print
    ref_max = 0
    for i in input0:
        if i > ref_max:
            ref_max = i

    errors = 0
    if output[0] != ref_max:
        print(f"Error: {output} != {ref_max}")
        errors += 1
    else:
        print(f"Correct output: {output} == {ref_max}")

    # If the result is correct, exit with a success code
    # Otherwise, exit with a failure code
    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print("\nError count: ", errors)
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
