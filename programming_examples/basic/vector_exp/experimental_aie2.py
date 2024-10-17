# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
from ml_dtypes import bfloat16

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.program import Program
from aie.api.worker import Worker
from aie.api.kernels.binkernel import BinKernel
from aie.api.phys.device import NPU1Col1
from aie.helpers.util import DataTiler
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
    a_fifos = A_fifo.second.split(
        [n * i for i in range(n_cores)], coords=(0, 1), types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.first.join(
        [n * i for i in range(n_cores)], coords=(0, 1), types=[tile_ty] * n_cores
    )

    io = IOCoordinator()
    with io.build_sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        for t in io.tile_loop(DataTiler(N)):
            io.fill(A_fifo.first, t, a_in, coords=(0, 0))
            io.drain(C_fifo.second, t, c_out, coords=(0, 0), wait=True)

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
                core_fn,
                fn_args=[a_fifos[i].second, c_fifos[i].first, exp_bf16_1024],
                coords=(0, 2 + i),
            )
        )

    return Program(NPU1Col1(), io, workers=workers)


my_eltwise_exp().resolve_program()
