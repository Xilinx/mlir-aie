#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.dialects.aiex import v8bfp16ebs8


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul():
    M = 64
    K = 128
    N = 64
    m = 64
    k = 128
    n = 64

    # Define tensor types
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]
    a_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    b_ty = np.ndarray[(k * n // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n // 8,), np.dtype[v8bfp16ebs8]]

    zero_kernel = Kernel(f"zero_kernel", f"mm_{m}x{k}x{n}.o", [c_ty])
    matmul_kernel = Kernel(
        "matmul_vectorized_bfp16",
        f"mm_{m}x{k}x{n}.o",
        [a_ty, b_ty, c_ty, np.int32, np.int32, np.int32],
    )

    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA")

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = None
    # b_dims = [(8, k // 8), (8, k), (k // 8, 1)]
    # b_dims = [(), (8, ), (N // 8, 8 * k // 8), (8, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")

    def core_fn(of_a, of_b, of_c, zero, matmul):
        elem_out = of_c.acquire(1)
        zero(elem_out)

        elem_in_a = of_a.acquire(1)
        elem_in_b = of_b.acquire(1)
        matmul(elem_in_a, elem_in_b, elem_out, m, k, n)
        of_a.release(1)
        of_b.release(1)

        of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xF00,
    )

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(inA.prod(), A)
        rt.fill(inB.prod(), B)
        rt.drain(outC.cons(), C, wait=True)

    dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    module = my_program.resolve_program(SequentialPlacer())
    return module


print(my_matmul())
