# vector_scalar_mul/vector_scalar_mul_iron_jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

import aie.iron as iron
from aie.iron import Compile, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


@iron.jit
def vector_scalar_mul(
    input0: iron.In,
    factor: iron.In,
    output: iron.Out,
    *,
    N: Compile[int],
    dtype: Compile[type] = np.int16,
    trace_size: Compile[int] = 0,
):
    num_sub_vectors = 4
    tile_size = N // num_sub_vectors

    # Define tensor types
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # External kernel from installed aie_kernels
    scale_fn = iron.kernels.scale(tile_size=tile_size, dtype=dtype)

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_factor = ObjectFifo(scalar_ty, name="infactor")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task for a compute tile to run
    def core_body(of_in, of_factor, of_out, scale_kernel):
        elem_factor = of_factor.acquire(1)
        for _ in range_(num_sub_vectors):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            scale_kernel(elem_in, elem_out, elem_factor, tile_size)
            of_in.release(1)
            of_out.release(1)
        of_factor.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale_fn],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (A, F, C):
        if trace_size:
            rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.fill(of_factor.prod(), F)
        rt.drain(of_out.cons(), C, wait=True)

    # Place components and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():
    N = 2048
    dtype = np.int16
    scale_factor = 3

    input0 = iron.randint(0, 100, (N,), dtype=dtype, device="npu")
    factor_tensor = iron.tensor([scale_factor], dtype=np.int32, device="npu")
    output = iron.zeros(N, dtype=dtype, device="npu")

    vector_scalar_mul(input0, factor_tensor, output, N=N, dtype=dtype)

    input0.to("cpu")
    output.to("cpu")
    expected = (input0.numpy().astype(np.int64) * scale_factor).astype(dtype)
    errors = np.sum(output.numpy() != expected)

    if not errors:
        print("\nPASS!\n")
        sys.exit(0)
    else:
        print(f"\nError count: {errors}")
        print("\nfailed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
