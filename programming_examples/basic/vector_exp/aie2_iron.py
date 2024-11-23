# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
from ml_dtypes import bfloat16

from aie.iron.io.iocoordinator import IOCoordinator
from aie.iron.dataflow.objectfifo import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.worker import Worker
from aie.iron.kernels.binkernel import BinKernel
from aie.iron.phys.device import NPU1Col1
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.ext.scf import _for as range_


def my_eltwise_exp():

    N = 65536

    # Tile sizes
    n = 1024
    N_div_n = N // n

    n_cores = 4
    tiles = N_div_n // n_cores
    buffer_depth = 2

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    memtile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    exp_bf16_1024 = BinKernel("exp_bf16_1024", "kernels.a", [tile_ty, tile_ty])

    A_fifo = ObjectFifo(buffer_depth, memtile_ty, "inA")
    C_fifo = ObjectFifo(buffer_depth, memtile_ty, "outC")
    a_fifos = A_fifo.cons.split(
        offsets=[n * i for i in range(n_cores)], types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.prod.join(
        offsets=[n * i for i in range(n_cores)], types=[tile_ty] * n_cores
    )

    tap = TensorTiler2D.simple_tiler((1, N))[0]

    io = IOCoordinator()
    with io.runtime_sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        io.fill(A_fifo.prod, tap, a_in)
        io.drain(C_fifo.cons, tap, c_out, wait=True)

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
            Worker(core_fn, fn_args=[a_fifos[i].cons, c_fifos[i].prod, exp_bf16_1024])
        )

    return Program(NPU1Col1(), io, workers=workers)


my_eltwise_exp().resolve_program(SequentialPlacer())
