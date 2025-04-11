# vector_exp/vector_exp.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_

if len(sys.argv) > 2:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))


def my_eltwise_exp():

    N = 65536

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 4
    tiles = N_div_n // n_cores

    # Define tensor types
    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    memtile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Generate handle to externally defined kernel function
    exp_bf16_1024 = Kernel("exp_bf16_1024", "kernels.a", [tile_ty, tile_ty])

    # Dataflow with ObjectFifos
    A_fifo = ObjectFifo(memtile_ty, name="inA")
    C_fifo = ObjectFifo(memtile_ty, name="outC")
    a_fifos = A_fifo.cons().split(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.prod().join(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )

    # Define a task a core might perform
    def core_fn(a_in, c_out, exp_bf16_1024):
        for _ in range_(tiles):
            elem_out = c_out.acquire(1)
            elem_in_a = a_in.acquire(1)
            exp_bf16_1024(elem_in_a, elem_out)
            a_in.release(1)
            c_out.release(1)

    # Create workers to run the tasks (one per core)
    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn, fn_args=[a_fifos[i].cons(), c_fifos[i].prod(), exp_bf16_1024]
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        rt.start(*workers)
        rt.fill(A_fifo.prod(), a_in)
        rt.drain(C_fifo.cons(), c_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


print(my_eltwise_exp())
