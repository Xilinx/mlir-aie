#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_


def vector_softmax(dev, trace_size):
    N = 262144  # *1024

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 2
    tiles = N_div_n // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the memory tile which aggregates across the 4 cores
    A_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    C_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]

    # AIE Core Function declarations
    softmax_bf16_vector = Kernel(
        "softmax_bf16", "kernels.a", [tile_ty, tile_ty, np.int32]
    )

    # AIE-array data movement with object fifos
    # Input A and Output C
    inA = ObjectFifo(A_memTile_ty, name="inA")
    outC = ObjectFifo(C_memTile_ty, name="outC")

    of_a_offsets = []
    of_c_offsets = []
    if n_cores > 1:
        of_a_offsets = [n * i for i in range(n_cores)]
        of_c_offsets = [n * i for i in range(n_cores)]
    inA_fifos = inA.cons().split(
        of_a_offsets,
        obj_types=[tile_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )
    outC_fifos = outC.prod().join(
        of_c_offsets,
        obj_types=[tile_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    # Task for the cores to perform
    def core_fn(of_in, of_out, softmax_kernel):
        for _ in range_(tiles):
            elem_out = of_out.acquire(1)
            elem_in_a = of_in.acquire(1)
            softmax_kernel(elem_in_a, elem_out, n)
            of_in.release(1)
            of_out.release(1)

    # Set up workers to perform the task
    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn,
                fn_args=[
                    inA_fifos[i].cons(),
                    outC_fifos[i].prod(),
                    softmax_bf16_vector,
                ],
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*workers)
        rt.fill(inA.prod(), A)
        rt.drain(outC.cons(), C, wait=True)

    # Place components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program()


try:
    device_name = str(sys.argv[1])
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))
    trace_size = 0 if (len(sys.argv) != 3) else int(sys.argv[2])
except ValueError:
    print("Argument is not an integer")

module = vector_softmax(dev, trace_size)
print(module)
