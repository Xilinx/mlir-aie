# passthrough_kernel/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.api.io.iocoordinator import IOCoordinator
from aie.api.dataflow.objectfifo import ObjectFifo
from aie.api.kernels.binkernel import BinKernel
from aie.api.placers import SequentialPlacer
from aie.api.program import Program
from aie.api.worker import Worker
from aie.api.phys.device import NPU1Col4
from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D, TensorTile
from aie.helpers.dialects.ext.scf import _for as range_


def my_matmul():
    M = 288
    K = 288
    m = 32
    k = 32

    # TODO: increase this
    n_cores = 1

    A_sz = M * K
    B_sz = K
    C_sz = M
    C_sz_div_n_cores = C_sz // n_cores

    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    # FIXME vectorized kernel is currently erroneous
    vectorized = False

    dtype_in = np.dtype[np.int16]
    dtype_in_str = "i16"
    dtype_out = np.dtype[np.int32]
    dtype_out_str = "i32"

    allA_ty = np.ndarray[(A_sz,), dtype_in]
    allB_ty = np.ndarray[(B_sz,), dtype_in]
    allC_ty = np.ndarray[(C_sz,), dtype_out]
    inA_ty = np.ndarray[(m * k,), dtype_in]
    inB_ty = np.ndarray[(k,), dtype_in]
    outC_ty = np.ndarray[(m,), dtype_out]
    A_ty = np.ndarray[(m, k), dtype_in]

    # AIE Core Function declarations
    func_type = "vectorized" if vectorized else "scalar"
    zero = BinKernel(f"zero_{func_type}_{dtype_out_str}", f"mv_{m}x{k}.o", [outC_ty])
    matvec = BinKernel(
        f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
        f"mv_{m}x{k}.o",
        [A_ty, inB_ty, outC_ty],
    )

    def core_fn(of_a, of_b, of_c, zero, matvec):
        for _ in range_(0xFFFFFFFF):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matvec(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    memA_fifos = []
    coreA_fifos = []
    outC_fifos = []
    workers = []
    B_fifo = ObjectFifo(2, inB_ty)
    for i in range(n_cores):
        a_fifo = ObjectFifo(2, inA_ty, f"memA{i}")
        memA_fifos.append(a_fifo)
        coreA_fifos.append(a_fifo.second.forward())  # TODO: transform if vectorized
        outC_fifos.append(ObjectFifo(2, outC_ty, f"outC{i}"))
        w = Worker(
            core_fn,
            [coreA_fifos[i].second, B_fifo.second, outC_fifos[i].first, zero, matvec],
        )
        workers.append(w)

    io = IOCoordinator()
    with io.build_sequence(allA_ty, allB_ty, allC_ty) as (a_in, b_in, c_out):
        A_tiler = TensorTiler2D(M, K, m, k)
        A_tile_iter = A_tiler.tile_iter(
            chunk_height=M_div_m_div_n_cores, chunk_width=K_div_k
        )

        C_tiler = TensorTiler2D(1, C_sz, 1, C_sz_div_n_cores)
        C_tile_iter = C_tiler.tile_iter()

        # TODO: don't support repeat yet so just make one tile
        b_tile = TensorTile(
            1, K, offset=0, sizes=[M_div_m_div_n_cores, 1, 1, K], strides=[0, 0, 0, 1]
        )
        io.fill(B_fifo.first, b_tile, b_in)

        for i, (a_tile, c_tile) in enumerate(io.tile_loop(A_tile_iter, C_tile_iter)):
            io.fill(memA_fifos[i].first, a_tile, a_in)
            io.drain(outC_fifos[i].second, c_tile, c_out, wait=True)

    my_program = Program(NPU1Col4(), io, workers)
    my_program.resolve_program(SequentialPlacer())


my_matmul()
