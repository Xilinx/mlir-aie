#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import numpy as np

from aie.extras.dialects.ext.scf import _for as range_
from aie.dialects.aiex import npu_dma_memcpy_nd, npu_sync

from aie.api.dataflow.inout.inout import MyInOutProgram
from aie.api.dataflow.objectfifo import MyObjectFifo
from aie.api.dataflow.objectfifolink import MyObjectFifoLink
from aie.api.kernels.binkernel import BinKernel
from aie.api.phys.device import NPU1Col4
from aie.api.program import MyProgram
from aie.api.worker import MyWorker

M = 288
K = 288
m = 32
k = 32

n_cores = 1

A_sz = M * K
B_sz = K
C_sz = M
C_sz_div_n_cores = C_sz // n_cores

M_div_m = M // m
M_div_m_div_n_cores = M // (m * n_cores)
K_div_k = K // k

m_x_k = m * k
m_x_K = m * K

# FIXME vectorized kernel is currently erroneous
vectorized = False

# Set dtypes
dtype_in = np.int16
dtype_in_str = "i16"
dtype_out = np.int32
dtype_out_str = "i32"

# Input/output tensor definitions # TODO: can simplify if single value?
inA_ty = np.ndarray(dtype_in, (M * K))
inB_ty = np.ndarray(dtype_in, (K,))
outC_ty = np.ndarray(dtype_out, (M,))
a_ty = np.ndarray[dtype_in, (m, k)]
a_flat_ty = np.ndarray[dtype_in, (m * k,)]
b_ty = np.ndarray[dtype_in, (k,)]
c_ty = np.ndarray[dtype_out, (m,)]

# AIE Core Function declarations
scalar_str = "" if vectorized else "scalar_"
zero = BinKernel(f"zero_{scalar_str}{dtype_out_str}", f"mv_{m}x{k}.o", [c_ty])
matvec = BinKernel(
    f"matvec_{scalar_str}{dtype_in_str}_{dtype_out_str}",
    f"mm_{m}x{k}x{n}.o",
    [a_ty, b_ty, c_ty],
)

memA_fifos = []
outC_fifos = []
A_links = []
worker_programs = []


# work to do on every core
def core_body(a_in, b_in, c_out, zero, matvec):
    for _ in range_(0xFFFFFFFF):
        elem_out = c_out.acquire(1)
        zero(elem_out)

        for _ in range_(K_div_k):
            elem_in_a = a_in.acquire(1)
            elem_in_b = b_in.acquire(1)

            matvec(elem_in_a, elem_in_b, elem_out)

            a_in.release(1)
            b_in.release(1)

        c_out.release(1)


# Setup workers + per-worker dataflow
inB_fifo = MyObjectFifo(2, b_ty, name="inB", end_first=(1, 0))
for i in range(n_cores):
    # Create object fifos for per-code dataflow
    memA = MyObjectFifo(2, a_flat_ty, name=f"memA{i}", end_first=(i, 0))
    toStreamA = [(k // 2 // 2, 2), (m, k), (2, 1)] if vectorized else []
    inA = MyObjectFifo(2, a_ty, name=f"inA{i}", toStream=toStreamA)
    outC = MyObjectFifo(2, c_ty, end_second=(i, 0))

    # Create per-core worker program
    worker_programs.append(
        MyWorker(
            core_body,
            [inA.second, inB_fifo.second, outC.first, zero, matvec],
            coords=(i, 2),
        )
    )

    # Save object fifos for later, keep in order
    A_links.append(MyObjectFifoLink(memA.second, inA.first, coords=(i, 1)))
    memA_fifos.append(memA)
    outC_fifos.append(outC)


# To/from AIE-array data movement
def sequence_fn(A, B, C, memA, inB, memC):
    npu_dma_memcpy_nd(
        metadata=inB.name,
        bd_id=2,
        mem=B,
        coords=(1, 0),
        sizes=[M_div_m_div_n_cores, 1, 1, K],
        strides=[0, 0, 0, 1],
    )
    for i in range(n_cores):
        A_offset = i * M_div_m_div_n_cores * m * K
        C_offset = i * M_div_m_div_n_cores * m
        npu_dma_memcpy_nd(
            metadata=memA[i].name,
            bd_id=1,
            mem=A,
            coords=(i, 0),
            offsets=[0, 0, 0, A_offset],
            sizes=[M_div_m_div_n_cores, K_div_k, m, k],
            strides=[m_x_K, k, K, 1],
        )
        npu_dma_memcpy_nd(
            metadata=memC[i].name,
            bd_id=0,
            mem=C,
            coords=(i, 0),
            offsets=[0, 0, 0, C_offset],
            sizes=[1, 1, 1, C_sz_div_n_cores],
            strides=[0, 0, 0, 1],
        )

    for i in range(n_cores):
        npu_sync(column=i, row=0, direction=0, channel=0)


inout_program = MyInOutProgram(
    sequence_fn,
    [inA_ty, inB_ty, outC_ty],
    [memA_fifos, inB_fifo, outC_fifos],
)

my_program = MyProgram(
    NPU1Col4(),
    worker_programs=worker_programs,
    links=A_links,
    inout_program=inout_program,
)

my_program.resolve_program()
