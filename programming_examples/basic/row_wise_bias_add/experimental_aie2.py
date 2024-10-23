# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.program import Program
from aie.api.kernels.binkernel import BinKernel
from aie.api.worker import Worker
from aie.api.phys.device import NPU1Col1
from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D
from aie.helpers.dialects.ext.scf import _for as range_


def row_wise_bias_add(M, N, m, n):

    assert M % m == 0
    assert N % n == 0

    tensor_ty = np.ndarray[(m * n,), np.dtype[np.float32]]
    bias_ty = np.ndarray[(n,), np.dtype[np.float32]]

    kernel_func = BinKernel(
        f"row_wise_bias_add_f32_f32", "kernel.o", [tensor_ty, bias_ty, tensor_ty]
    )

    in_fifo = ObjectFifo(2, tensor_ty, "in_fifo")
    bias_fifo = ObjectFifo(2, bias_ty, "bias_fifo")
    out_fifo = ObjectFifo(2, tensor_ty, "out_fifo")

    def core_fn(in_fifo, bias_fifo, out_fifo, kernel_func):
        for _ in range_(0xFFFFFFFF):
            for _ in range_(N // n):
                elem_bias = bias_fifo.acquire(1)
                for _ in range_(M // m):
                    elem_in = in_fifo.acquire(1)
                    elem_out = out_fifo.acquire(1)
                    kernel_func(elem_in, elem_bias, elem_out)
                    out_fifo.release(1)
                    in_fifo.release(1)
                bias_fifo.release(1)

    io = IOCoordinator()
    with io.build_sequence(tensor_ty, bias_ty, tensor_ty) as (inp, bias, out):
        tiler = TensorTiler2D(M, N, m, n, tensor_col_major=True)
        bias_tiler = TensorTiler2D(1, N, 1, n)
        for t, bias_t in io.tile_loop(
            tiler.tile_iter(chunk_height=M // m, chunk_width=N // n),
            bias_tiler.tile_iter(chunk_width=N // n),
        ):
            io.fill(in_fifo.first, t, inp, coords=(0, 0))
            io.fill(bias_fifo.first, bias_t, bias, coords=(0, 0))
            io.drain(out_fifo.second, t, out, coords=(0, 0), wait=True)

    my_worker = Worker(
        core_fn,
        fn_args=[in_fifo.second, bias_fifo.second, out_fifo.first, kernel_func],
        coords=(0, 2),
    )
    return Program(NPU1Col1(), io, workers=[my_worker])


program = row_wise_bias_add(
    int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
program.resolve_program()
