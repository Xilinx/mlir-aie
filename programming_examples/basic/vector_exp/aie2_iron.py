# vector_exp/aie2_iron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
from ml_dtypes import bfloat16

from aie.iron.runtime import Runtime
from aie.iron.dataflow import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.worker import Worker
from aie.iron.kernels import BinKernel
from aie.iron.phys.device import NPU1Col1
from aie.helpers.dialects.ext.scf import _for as range_


def my_eltwise_exp():

    N = 65536

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 4
    tiles = N_div_n // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    memtile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    exp_bf16_1024 = BinKernel("exp_bf16_1024", "kernels.a", [tile_ty, tile_ty])

    A_fifo = ObjectFifo(memtile_ty, name="inA")
    C_fifo = ObjectFifo(memtile_ty, name="outC")
    a_fifos = A_fifo.cons().split(
        offsets=[n * i for i in range(n_cores)], types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.prod().join(
        offsets=[n * i for i in range(n_cores)], types=[tile_ty] * n_cores
    )

    def core_fn(a_in, c_out, exp_bf16_1024):
        for _ in range_(0xFFFFFFFF):
            for _ in range_(tiles):
                elem_out = c_out.acquire(1)
                elem_in_a = a_in.acquire(1)
                exp_bf16_1024(elem_in_a, elem_out)
                a_in.release(1)
                c_out.release(1)

    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn, fn_args=[a_fifos[i].cons(), c_fifos[i].prod(), exp_bf16_1024]
            )
        )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        rt.start(*workers)
        rt.fill(A_fifo.prod(), a_in)
        rt.drain(C_fifo.cons(), c_out, wait=True)

    return Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())


print(my_eltwise_exp())