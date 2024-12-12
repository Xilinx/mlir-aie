# vector_reduce_max/vector_reduce_max.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D


def my_reduce_max():
    N = 512
    C = 1 # FIXME breaks if C > 1

    # Define tensor types
    in_ty = np.ndarray[(N, N,), np.dtype[np.int32]]
    out_ty = np.ndarray[(N,), np.dtype[np.int32]]

    # Define tensor tile types
    in_tile_ty = np.ndarray[(C, N,), np.dtype[np.int32]]
    out_tile_ty = np.ndarray[(C,), np.dtype[np.int32]]

    # Define worker tensor tile types
    in_worker_ty = np.ndarray[(N,), np.dtype[np.int32]]
    out_worker_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(in_tile_ty, name="in")
    of_out = ObjectFifo(out_tile_ty, name="out")
    of_ins = of_in.cons().split([i * N for i in range(C)], obj_types=[in_worker_ty] * C)
    of_outs = of_out.prod().join([i for i in range(C)], obj_types=[out_worker_ty] * C)

    # AIE Core Function declarations
    reduce_add_vector = Kernel(
        "reduce_max_vector", "reduce_max.cc.o", [in_worker_ty, out_worker_ty, np.int32]
    )

    # Define a task to run
    def core_body(of_in, of_out, reduce_add_vector):
        elem_out = of_out.acquire(1)
        elem_in = of_in.acquire(1)
        reduce_add_vector(elem_in, elem_out, N)
        of_in.release(1)
        of_out.release(1)

    # Define a worker to run the task on a core
    workers = [Worker(core_body, fn_args=[of_ins[i].cons(), of_outs[i].prod(), reduce_add_vector]) for i in range(C)]
    
    # Tile the input matrix for a row at a time
    taps_in = TensorTiler2D.simple_tiler((N, N), (N, N))
    taps_out = TensorTiler2D.simple_tiler((N,), (N,))

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(*workers)
        for t_i, t_o in zip(taps_in, taps_out):
            tg = rt.task_group()
            rt.fill(of_in.prod(), a_in, t_i, task_group=tg)
            rt.drain(of_out.cons(), c_out, t_o, task_group=tg, wait=True)
            rt.finish_task_group(tg)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())


print(my_reduce_max())
