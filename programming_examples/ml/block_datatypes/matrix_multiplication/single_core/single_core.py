#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np

from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.placers import SequentialPlacer


def ceildiv(a, b):
    return (a + b - 1) // b


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core) with bfp16ebs8 input/output",
        description="Emits MLIR code for a matrix multiplication design of the given input size. Only supported in NPU2 devices.",
    )
    argparser.add_argument("-M", type=int, default=128)
    argparser.add_argument("-K", type=int, default=128)
    argparser.add_argument("-N", type=int, default=128)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=64)
    args = argparser.parse_args()
    print(my_matmul(args.M, args.K, args.N, args.m, args.k, args.n))


def my_matmul(M, K, N, m, k, n):
    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Define tensor types
    A_ty = np.ndarray[(M * K // 8,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N // 8,), np.dtype[v8bfp16ebs8]]
    a_ty = np.ndarray[(m, k // 8), np.dtype[v8bfp16ebs8]]
    b_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    c_ty = np.ndarray[(m, n // 8), np.dtype[v8bfp16ebs8]]

    zero_kernel = Kernel(f"zero_kernel", f"mm_{m}x{k}x{n}.o", [c_ty])
    matmul_kernel = Kernel(
        "matmul_vectorized_bfp16",
        f"mm_{m}x{k}x{n}.o",
        [a_ty, b_ty, c_ty],
    )

    inA = ObjectFifo(a_ty, name="inA")
    memA = inA.cons().forward(name="memA")

    inB = ObjectFifo(b_ty, name="inB")
    b_dims = None
    # This would be equivalent to a transposition of a 64x64 pre-shuffled tile
    # b_dims = [(8, 8), (8, 64), (8, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    outC = memC.cons().forward(name="outC")

    def core_fn(of_a, of_b, of_c, zero, matmul):
        for _ in range_(tiles) if tiles > 1 else range(1):
            elem_out = of_c.acquire(1)
            zero(elem_out)

            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)

            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), zero_kernel, matmul_kernel],
        stack_size=0xD00,
    )

    rows_per_block = 4

    A_tiles = TensorTiler2D.group_tiler(
        (M, K // 8), (m, k // 8), (1, K_div_k), pattern_repeat=N_div_n
    )
    b_tap = TensorTiler2D.group_tiler((N, K // 8), (n, k // 8), (N_div_n, K_div_k))[0]

    C_tiles = TensorTiler2D.group_tiler(
        (M, N // 8), (m, n // 8), (rows_per_block // 2, N_div_n)
    )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)
        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:

                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])

                    rt.fill(inB.prod(), B, tap=b_tap, task_group=tgs[-1])

                rt.drain(
                    outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True
                )
                c_index += 1

                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]

        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    module = my_program.resolve_program(SequentialPlacer())
    return module


main()
