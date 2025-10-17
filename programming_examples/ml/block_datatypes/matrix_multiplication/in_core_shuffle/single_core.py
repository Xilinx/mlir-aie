#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.localbuffer import LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.dialects.aiex import v8bfp16ebs8


def ceildiv(a, b):
    return (a + b - 1) // b


def main():
    argparser = argparse.ArgumentParser(
        prog="Example of shuffling using the scalar unit inside a core",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    args = argparser.parse_args()
    print(my_matmul(args.M, args.K, args.N, args.m, args.k, args.n))


def my_matmul(M, K, N, m, k, n):

    # Define tensor types
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]
    a_ty = np.ndarray[(m * k // 8,), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m * n // 8,), np.dtype[v8bfp16ebs8]]

    scalar_shuffle_kernel = Kernel(
        "scalar_shuffle",
        f"mm_{m}x{k}x{n}.o",
        [a_ty, c_ty, np.int16, np.int16, np.int16],
    )

    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA")

    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")

    def core_fn(of_a, of_c, scalar_shuffle_kernel):
        elem_out = of_c.acquire(1)
        elem_in_a = of_a.acquire(1)
        # Note that it is possible to use a buffer here to
        # do the shuffling in instead:
        # buffer = LocalBuffer(a_ty)
        scalar_shuffle_kernel(elem_in_a, elem_out, k, m, False)
        of_a.release(1)
        of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memC.prod(), scalar_shuffle_kernel],
        stack_size=0xF00,
    )

    # Note that B is completely unused in this design, will have to be removed in the future
    rt = Runtime()
    with rt.sequence(A_ty, C_ty) as (A, C):
        rt.start(worker)
        rt.fill(inA.prod(), A)
        rt.drain(outC.cons(), C, wait=True)

    dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    module = my_program.resolve_program(SequentialPlacer())
    return module


main()
