# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

devices = {
    "npu": NPU1(),
    "npu2": NPU2()
}
if len(sys.argv) != 2 or sys.argv[1] not in devices:
    print(f"Usage {sys.argv[0]} <{'|'.join(devices.keys())}>")
    sys.exit(1)
device = devices[sys.argv[1]]

M, K, N = 32, 32, 32  # Problem size
m, k, n = 16, 16, 16  # Tile size
r, s, t = 8, 2, 8  # Intrinsic size


# --------------------------------------------------------------------------
# In-Array Data Movement
# --------------------------------------------------------------------------

A_ty = np.ndarray[(M, K), np.dtype[np.int16]]
B_ty = np.ndarray[(K, N), np.dtype[np.int16]]
C_ty = np.ndarray[(M, N), np.dtype[np.int16]]
a_ty = np.ndarray[(m, k), np.dtype[np.int16]]
b_ty = np.ndarray[(k, n), np.dtype[np.int16]]
c_ty = np.ndarray[(m, n), np.dtype[np.int16]]

# The following ObjectFIFOs route m*k-, k*n-, and m*n-sized subtiles
# (objects) to/from the compute cores via mem tiles, rearranging their data
# into r*s-, s*t-, and r*t-sized sub-subtiles.

fifo_A_L3L2 = ObjectFifo(a_ty, name="A_L3L2")
tap_A_L2L1 = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
fifo_A_L2L1 = fifo_A_L3L2.cons().forward(
    dims_to_stream=tap_A_L2L1.transformation_dims, 
    name="A_L2L1"
)

fifo_B_L3L2 = ObjectFifo(b_ty, name="B_L3L2")
tap_B_L2L1 = TensorTiler2D.group_tiler((k, n), (s, t), (k // s, n // t))[0]
fifo_B_L2L1 = fifo_B_L3L2.cons().forward(
    dims_to_stream=tap_B_L2L1.transformation_dims, 
    name="B_L2L1"
)

fifo_C_L1L2 = ObjectFifo(c_ty, name="C_L1L2")
tap_C_L1L2 = TensorAccessPattern(
    tensor_dims=(m, n),
    offset=0,
    sizes=[m // r, r, n // t, t],
    strides=[r * n, t, r * t, 1]
)
fifo_C_L2L3 = fifo_C_L1L2.cons().forward(
    dims_to_stream=tap_C_L1L2.transformation_dims, 
    name="C_L2L3"
)


# --------------------------------------------------------------------------
# Task each core will run
# --------------------------------------------------------------------------

# The kernel repeatedly acquires one subtile of A and B, multiplies them,
# and accumulates the result on top of C. As these tiles come in, the DMAs
# will have rearranged them into r*s-, s*t-, and r*t-sized subtiles, which
# the computation kernel relies on.

zero_kernel = Kernel("zero", f"matrix_multiplication.o", [c_ty])
matmul_kernel = Kernel("matrix_multiplication", f"matrix_multiplication.o", [a_ty, b_ty, c_ty])
def core_fn(of_a, of_b, of_c, zero, matmul):
    for _ in range_(M // m * N // n):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range_(K // k):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            matmul(elem_in_a, elem_in_b, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)
worker = Worker(core_fn, [fifo_A_L2L1.cons(), fifo_B_L2L1.cons(), fifo_C_L1L2.prod(), zero_kernel, matmul_kernel])


# --------------------------------------------------------------------------
# DRAM-NPU data movement and work dispatch
# --------------------------------------------------------------------------

# The data movement patterns from DRAM divide the input matrices (sizes 
# M*K, K*N) into m*k- and k*n-sized subtiles and produce output into C in
# m*n-sized subtiles. Each single "task group" encompasses all data
# movement required for a single row of the output matrix.

a_taps = TensorTiler2D.group_tiler((M, K), (m, k), (1, K // k), pattern_repeat=(N // n))
b_tap = TensorTiler2D.group_tiler((K, N), (k, n), (K // k, N // n), tile_group_col_major=True)[0]
c_taps = TensorTiler2D.group_tiler((M, N), (m, n), (1, N // n))

rt = Runtime()
with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
    rt.start(worker)
    for tile_row in range(M // m):
        task_group = rt.task_group()
        rt.fill(fifo_A_L3L2.prod(), A, tap=a_taps[tile_row], task_group=task_group)
        rt.fill(fifo_B_L3L2.prod(), B, tap=b_tap, task_group=task_group)
        rt.drain(fifo_C_L2L3.cons(), C, tap=c_taps[tile_row], task_group=task_group, wait=True)
        rt.finish_task_group(task_group)


# --------------------------------------------------------------------------
# Place and generate MLIR program
# --------------------------------------------------------------------------

program = Program(device, rt)
module = program.resolve_program(SequentialPlacer())
print(module)
