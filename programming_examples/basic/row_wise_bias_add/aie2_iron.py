# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron.io.iocoordinator import IOCoordinator
from aie.iron.dataflow.objectfifo import ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.program import Program
from aie.iron.kernels.binkernel import BinKernel
from aie.iron.worker import Worker
from aie.iron.phys.device import NPU1Col1
from aie.helpers.taplib import TensorTiler2D
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
        for _ in range_(N // n):
            elem_bias = bias_fifo.acquire(1)
            for _ in range_(M // m):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                kernel_func(elem_in, elem_bias, elem_out)
                out_fifo.release(1)
                in_fifo.release(1)
            bias_fifo.release(1)

    tiler = TensorTiler2D.group_tiler(
        (M, N), (m, n), (M // m, N // n), tile_group_col_major=True
    )
    bias_tiler = TensorTiler2D.group_tiler((1, N), (1, n), (1, N // n))

    io = IOCoordinator()
    with io.runtime_sequence(tensor_ty, bias_ty, tensor_ty) as (inp, bias, out):
        io.fill(in_fifo.prod, tiler[0], inp)
        io.fill(bias_fifo.prod, bias_tiler[0], bias)
        io.drain(out_fifo.cons, tiler[0], out, wait=True)

    my_worker = Worker(
        core_fn,
        fn_args=[in_fifo.cons, bias_fifo.cons, out_fifo.prod, kernel_func],
        while_true=True,
    )
    return Program(NPU1Col1(), io, workers=[my_worker])


program = row_wise_bias_add(
    int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
program.resolve_program(SequentialPlacer())
