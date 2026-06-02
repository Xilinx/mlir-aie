#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
import argparse
import numpy as np

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.aiex import v8bfp16ebs8
import aie.utils.trace as trace_utils
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.dialects.scf import _for as range_


def ceildiv(a, b):
    return (a + b - 1) // b


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (32 Cores, 8x4 grid) with bfp16ebs8 weights and bf16 input/output",
        description="Emits MLIR code for a matrix multiplication design of the given input size using 32 cores (8 columns x 4 rows). Only supported in NPU2 devices.",
    )
    argparser.add_argument("-M", type=int, default=0)
    argparser.add_argument("-K", type=int, default=0)
    argparser.add_argument("-N", type=int, default=0)
    argparser.add_argument("-m", type=int, default=0)
    argparser.add_argument("-k", type=int, default=0)
    argparser.add_argument("-n", type=int, default=0)
    args = argparser.parse_args()
    with mlir_mod_ctx() as ctx:
        my_matmul(args.M, args.K, args.N, args.m, args.k, args.n)
        print(ctx.module)


def my_matmul(M, K, N, m, k, n):

    DIV = 4
    n_aie_cols = 8
    n_aie_rows = 4

    # L1 tile sizes (compute level)
    a_m_l1 = m  # for A
    a_k_l1 = k  # for A
    b_k_l1 = k  # for B
    b_n_l1 = n  # for B
    c_m_l1 = m  # for C
    c_n_l1 = n  # for C

    a_m_l2 = m  # for A
    a_k_l2 = k  # for A
    b_k_l2 = k  # for B
    b_n_l2 = n  # for B
    c_m_l2 = n_aie_rows * m  # for C
    c_n_l2 = n  # for C

    r = 8
    s = 8
    t = 8

    enable_tracing = False
    trace_size = 8192

    # Use bfloat16 for input/output and bfp16ebs8 for weights
    dtype_in = v8bfp16ebs8
    dtype_out = v8bfp16ebs8

    dev_ty = AIEDevice.npu2

    @device(dev_ty)
    def device_body():
        A_l2_ty = np.ndarray[(m, k // 8), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[
            (k, n // 8), np.dtype[v8bfp16ebs8]
        ]  # Full B matrix in memory
        C_l2_ty = np.ndarray[(n_aie_rows * m, n // 8), np.dtype[dtype_out]]

        A_l1_ty = np.ndarray[(m // DIV, k // 8), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[
            (k, n // 8), np.dtype[v8bfp16ebs8]
        ]  # Use v8bfp16ebs8 for weights
        C_l1_ty = np.ndarray[(m, n // 8), np.dtype[dtype_out]]

        # AIE Core Function declarations
        zero = external_func(
            f"zero_kernel", inputs=[C_l1_ty], link_with=f"mm_{m}x{k}x{n}.o"
        )
        matmul = external_func(
            "matmul_vectorized_bfp16",
            inputs=[A_l1_ty, B_l1_ty, C_l1_ty],
            link_with=f"mm_{m}x{k}x{n}.o",
        )

        # Tile declarations as tile[row][col]
        # using columns 0-7
        tiles = [
            [tile(col, row) for col in range(0, n_aie_cols)]
            for row in range(0, n_aie_rows + 2)
        ]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        shim_tile_trace = tile(1, 0)
        tiles_to_trace = [
            tile(0, 1),
            tile(0, 2),
            tile(0, 0),
        ]  # Last compute tile and last mem tile
        if enable_tracing:
            trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile_trace)

        # AIE-array data movement with object fifos
        A_l3l2_fifos = [None] * n_aie_rows
        A_l2l1_fifos = [None] * n_aie_rows

        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols

        C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
        C_l2l3_fifos = [None] * n_aie_cols

        # Input A
        for row in range(4):
            A_transformations = []  # [(m // r, r * k),(k // s, s),(r, k),(s, 1),]
            A_l3l2_fifos[row] = object_fifo(
                f"A_L3L2_{row}",
                shim_tiles[2 * row],
                mem_tiles[2 * row],
                2,
                A_l2_ty,
                None,
            )

            A_l2l1_fifos[row] = object_fifo(
                f"A_L2L1_{row}",
                mem_tiles[2 * row],
                core_tiles[row][0:8],
                2,
                A_l1_ty,
                A_transformations,
            )
            object_fifo_link(A_l3l2_fifos[row], A_l2l1_fifos[row])

        # Input B
        for col in range(8):
            B_transformations = []  # [(n // t, t * k),(k // s, s),(t, k),(s, 1),]
            B_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                2,
                B_l2_ty,
            )
            B_l2l1_fifos[col] = object_fifo(
                f"B_L2L1_{col}",
                mem_tiles[col],
                [core_tiles[j][col] for j in range(4)],
                2,
                B_l1_ty,
                B_transformations,
            )
            object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])

        # Output C
        for col in range(n_aie_cols):
            for row in range(n_aie_rows):
                C_l1l2_fifos[row][col] = object_fifo(
                    f"C_L1L2_{col}_{row}",
                    core_tiles[row][col],
                    mem_tiles[col],
                    1,
                    C_l1_ty,
                )
            C_transformations = []  # [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
            C_l2l3_fifos[col] = object_fifo(
                f"C_L2L3_{col}",
                mem_tiles[col],
                shim_tiles[col],
                2,
                C_l2_ty,
                C_transformations,
            )
            of_offsets = [m * n // 8 * i for i in range(n_aie_rows)]
            object_fifo_link(
                [C_l1l2_fifos[j][col] for j in range(n_aie_rows)],
                C_l2l3_fifos[col],
                of_offsets,
            )

        # Set up compute tiles
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):

                @core(core_tiles[row][col], stack_size=0xF00)  # 0xF00?
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range(
                            (M // m) * (N // n) // (n_aie_cols * n_aie_rows)
                        ):
                            elem_out = C_l1l2_fifos[row][col].acquire(
                                ObjectFifoPort.Produce, 1
                            )
                            zero(elem_out)
                            for _ in range_(K // k):
                                elem_in_b = B_l2l1_fifos[col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                for i in range(DIV):
                                    elem_in_a = A_l2l1_fifos[row].acquire(
                                        ObjectFifoPort.Consume, 1
                                    )
                                    matmul(elem_in_a, elem_in_b, elem_out)
                                    A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                                B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)
                            C_l1l2_fifos[row][col].release(ObjectFifoPort.Produce, 1)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[(M * K // 8,), np.dtype[dtype_in]],
            np.ndarray[
                (K * N // 8,), np.dtype[v8bfp16ebs8]
            ],  # B0 for core1 (odd columns)
            np.ndarray[(M * N // 8,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):

            if enable_tracing:
                trace_utils.configure_packet_tracing_aie2(
                    tiles_to_trace,
                    shim_tile_trace,
                    trace_size,
                )
            # flat input (not correct, buf faster)
            A_taps = TensorTiler2D.group_tiler((1, M * K // 8), (1, m * K // 8), (1, 1))
            B_taps = TensorTiler2D.group_tiler((1, N * K // 8), (1, n * K // 8), (1, 1))
            C_taps = TensorTiler2D.group_tiler(
                (1, M * N // 8), (1, n_aie_rows * m * n // 8), (1, 1)
            )
            # Multi-group task lists for managing DMA IDs
            num_row_tile = M // m // n_aie_rows
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile
            tb_max_n_rows = 4
            input_task_groups = [[] for _ in range(tb_max_n_rows)]
            output_task_groups = [[] for _ in range(tb_max_n_rows)]

            # Create tasks for all groups using loops
            for group_idx in range(num_groups):
                # Create A tasks for this group (4 tasks per group)
                a_base_idx = (group_idx // num_col_tile) * n_aie_rows
                for row in range(n_aie_rows):
                    a_task = shim_dma_single_bd_task(
                        A_l3l2_fifos[row],
                        A,
                        tap=A_taps[a_base_idx + row],
                        issue_token=False,
                    )
                    dma_start_task(a_task)
                    input_task_groups[group_idx % tb_max_n_rows].append(a_task)

                # Create B tasks for this group (8 tasks per group)
                b_base_idx = (group_idx % num_col_tile) * n_aie_cols
                for col in range(n_aie_cols):
                    b_task = shim_dma_single_bd_task(
                        B_l3l2_fifos[col],
                        B,
                        tap=B_taps[b_base_idx + col],
                        issue_token=False,
                    )
                    dma_start_task(b_task)
                    input_task_groups[group_idx % tb_max_n_rows].append(b_task)

                # Create C output tasks for this group (8 tasks per group)
                c_base_idx = group_idx * 8
                for col in range(n_aie_cols):
                    c_task = shim_dma_single_bd_task(
                        C_l2l3_fifos[col],
                        C,
                        tap=C_taps[c_base_idx + col],
                        issue_token=True,
                    )
                    dma_start_task(c_task)
                    output_task_groups[group_idx % tb_max_n_rows].append(c_task)
                if (group_idx % tb_max_n_rows == 1) and (group_idx != 1):
                    dma_await_task(*output_task_groups[2])
                    output_task_groups[2] = []
                    dma_free_task(*input_task_groups[2])
                    input_task_groups[2] = []
                    dma_await_task(*output_task_groups[3])
                    output_task_groups[3] = []
                    dma_free_task(*input_task_groups[3])
                    input_task_groups[3] = []
                if group_idx % tb_max_n_rows == 3:
                    dma_await_task(*output_task_groups[0])
                    output_task_groups[0] = []
                    dma_free_task(*input_task_groups[0])
                    input_task_groups[0] = []
                    dma_await_task(*output_task_groups[1])
                    output_task_groups[1] = []
                    dma_free_task(*input_task_groups[1])
                    input_task_groups[1] = []

            dma_await_task(*output_task_groups[2])
            dma_free_task(*input_task_groups[2])
            dma_await_task(*output_task_groups[3])
            dma_free_task(*input_task_groups[3])

            if enable_tracing:
                trace_utils.gen_trace_done_aie2(shim_tile_trace)


main()
