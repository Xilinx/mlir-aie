#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.
#
# Dynamic single-core GEMM design with placed runtime sequence (bf16->f32, 32x32x32 tiles).
#
# Uses the dynamic core body (RTP reads + scf.while loops) for compile-once-run-any-size,
# combined with the placed runtime sequence pattern (shim_dma_single_bd_task / dma_start_task
# / dma_await_task / dma_free_task) instead of npu_dma_memcpy_nd.

import argparse
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import WhileOp, condition, yield_
from aie.dialects.arith import (
    constant as arith_constant,
    CmpIOp,
    CmpIPredicate,
    AddIOp,
    IndexCastOp,
)
from aie.dialects.memref import LoadOp
from aie.extras.dialects.arith import constant
import aie.utils.trace as trace_utils
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype
from aie.ir import IndexType, IntegerType, InsertionPoint
from aie.extras import types as T


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Dynamic Matrix Multiplication MLIR Design (Single Core, Placed)",
        description="Emits MLIR code for a dynamic matrix multiplication design with placed runtime sequence",
    )
    argparser.add_argument("--dev", type=str, choices=["npu2"], default="npu2")
    argparser.add_argument("-M", type=int, default=128)
    argparser.add_argument("-K", type=int, default=128)
    argparser.add_argument("-N", type=int, default=128)
    argparser.add_argument("--dtype_in", type=str, choices=["bf16"], default="bf16")
    argparser.add_argument("--dtype_out", type=str, choices=["f32"], default="f32")
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.dtype_in,
        args.dtype_out,
        args.trace_size,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(dev, M, K, N, dtype_in_str, dtype_out_str, trace_size):
    # Fixed tile sizes for dynamic design
    m, k, n = 32, 32, 32

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    # NPU2 bf16 microkernel dimensions: r=4, s=8, t=8
    r, s, t = 4, 8, 8

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    A_sz = M * K
    B_sz = K * N
    C_sz = M * N

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    enable_tracing = trace_size > 0

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
            b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
            c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

            # AIE Core Function declarations
            zero = external_func(f"zero_{dtype_out_str}", inputs=[c_ty])
            matmul = external_func(
                f"matmul_{dtype_in_str}_{dtype_out_str}",
                inputs=[a_ty, b_ty, c_ty],
            )

            # Tile declarations
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile2_col, compute_tile2_row = 0, 2
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            # Input A
            inA = object_fifo("inA", shim_tile, mem_tile, 2, a_ty)
            memA = object_fifo(
                "memA",
                mem_tile,
                compute_tile2,
                2,
                a_ty,
                [
                    (m // r, r * k),
                    (k // s, s),
                    (r, k),
                    (s, 1),
                ],
            )
            object_fifo_link(inA, memA)

            # Input B
            inB = object_fifo("inB", shim_tile, mem_tile, 2, b_ty)
            memB = object_fifo(
                "memB",
                mem_tile,
                compute_tile2,
                2,
                b_ty,
                [
                    (k // s, s * n),
                    (n // t, t),
                    (s, n),
                    (t, 1),
                ],
            )
            object_fifo_link(inB, memB)

            # Output C
            memC = object_fifo("memC", compute_tile2, mem_tile, 2, c_ty)
            outC = object_fifo(
                "outC",
                mem_tile,
                shim_tile,
                2,
                c_ty,
                [
                    (m // r, r * n),
                    (r, t),
                    (n // t, r * t),
                    (t, 1),
                ],
            )
            object_fifo_link(memC, outC)

            # RTP buffer: [0] = K_div_k, [1] = tiles (M_div_m * N_div_n)
            rtp_buf = buffer(
                compute_tile2,
                T.memref(16, T.i32()),
                name="rtp",
                address=0x600,
            )

            # Set up tracing
            tiles_to_trace = [compute_tile2]
            if enable_tracing:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            # Core body with dynamic loop bounds via RTP
            @core(
                compute_tile2,
                f"mm_{m}x{k}x{n}.o",
                stack_size=0xD00,
                dynamic_objfifo_lowering=True,
            )
            def core_body():
                i32_ty = IntegerType.get_signless(32)
                idx_ty = IndexType.get()

                c0 = arith_constant(i32_ty, 0)
                c1 = arith_constant(i32_ty, 1)
                c0_idx = constant(0, index=True)
                c1_idx = constant(1, index=True)
                cmax_idx = constant(0xFFFFFFFF, index=True)

                # Infinite outer loop
                for _ in range_(c0_idx, cmax_idx, c1_idx):

                    # Read tiles count from RTP[1]
                    tiles_i32 = LoadOp(rtp_buf, [c1_idx]).result
                    tiles_idx = IndexCastOp(idx_ty, tiles_i32).result

                    # Tile loop (scf.while for dynamic bounds)
                    tile_while = WhileOp([idx_ty], [c0_idx])

                    # "before" region: check condition
                    before_block = tile_while.before.blocks.append(idx_ty)
                    with InsertionPoint(before_block):
                        tile_iter = before_block.arguments[0]
                        tile_cond = CmpIOp(
                            CmpIPredicate.slt, tile_iter, tiles_idx
                        ).result
                        condition(tile_cond, [tile_iter])

                    # "after" region: loop body
                    after_block = tile_while.after.blocks.append(idx_ty)
                    with InsertionPoint(after_block):
                        tile_iter = after_block.arguments[0]

                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)

                        # Read K_div_k from RTP[0]
                        k_iters_i32 = LoadOp(rtp_buf, [c0_idx]).result
                        k_iters_idx = IndexCastOp(idx_ty, k_iters_i32).result

                        # K accumulation loop (scf.while)
                        k_while = WhileOp([idx_ty], [c0_idx])

                        k_before = k_while.before.blocks.append(idx_ty)
                        with InsertionPoint(k_before):
                            k_iter = k_before.arguments[0]
                            k_cond = CmpIOp(
                                CmpIPredicate.slt, k_iter, k_iters_idx
                            ).result
                            condition(k_cond, [k_iter])

                        k_after = k_while.after.blocks.append(idx_ty)
                        with InsertionPoint(k_after):
                            k_iter = k_after.arguments[0]

                            elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_a, elem_in_b, elem_out)
                            memA.release(ObjectFifoPort.Consume, 1)
                            memB.release(ObjectFifoPort.Consume, 1)

                            k_next = AddIOp(k_iter, c1_idx).result
                            yield_([k_next])

                        memC.release(ObjectFifoPort.Produce, 1)

                        tile_next = AddIOp(tile_iter, c1_idx).result
                        yield_([tile_next])

            # Placed runtime_sequence using shim_dma_single_bd_task
            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
            )
            def sequence(A, B, C):
                if enable_tracing:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace=tiles_to_trace,
                        shim=shim_tile,
                        trace_size=trace_size,
                        coretile_events=[
                            trace_utils.events.PortEvent(
                                trace_utils.events.CoreEvent.PORT_RUNNING_0,
                                port_number=1,
                                master=True,
                            ),
                            trace_utils.events.PortEvent(
                                trace_utils.events.CoreEvent.PORT_RUNNING_1,
                                port_number=2,
                                master=True,
                            ),
                            trace_utils.events.PortEvent(
                                trace_utils.events.CoreEvent.PORT_RUNNING_2,
                                port_number=1,
                                master=False,
                            ),
                            trace_utils.events.CoreEvent.INSTR_EVENT_0,
                            trace_utils.events.CoreEvent.INSTR_EVENT_1,
                            trace_utils.events.CoreEvent.MEMORY_STALL,
                            trace_utils.events.CoreEvent.LOCK_STALL,
                            trace_utils.events.CoreEvent.INSTR_VECTOR,
                        ],
                    )

                # Write RTP values for the static compilation size
                npu_rtp_write("rtp", 0, K_div_k)
                npu_rtp_write("rtp", 1, tiles)

                # Use 2 tile rows per block (placed style, prevents BD exhaustion)
                rows_per_block = 2

                # Define tensor access patterns using TensorTiler2D
                A_taps = TensorTiler2D.group_tiler(
                    (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n
                )
                b_tap = TensorTiler2D.group_tiler(
                    (K, N),
                    (k, n),
                    (K_div_k, N_div_n),
                    tile_group_col_major=True,
                )[0]
                C_taps = TensorTiler2D.group_tiler(
                    (M, N), (m, n), (rows_per_block // 2, N_div_n)
                )
                c_index = 0

                a_tasks = []
                b_tasks = []
                c_tasks = []

                for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                    for pingpong in [0, 1]:
                        row_base = (
                            tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2
                        )
                        num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                        if num_tile_rows <= 0:
                            break

                        # -- C --
                        c_task = shim_dma_single_bd_task(
                            outC,
                            C,
                            tap=C_taps[c_index],
                            issue_token=True,
                        )
                        c_index += 1
                        dma_start_task(c_task)
                        c_tasks.append(c_task)

                        for tile_row in range(num_tile_rows):
                            # -- A --
                            tile_offset = (row_base + tile_row) % len(A_taps)
                            a_task = shim_dma_single_bd_task(
                                inA, A, tap=A_taps[tile_offset]
                            )
                            dma_start_task(a_task)
                            a_tasks.append(a_task)

                            # -- B --
                            b_task = shim_dma_single_bd_task(inB, B, tap=b_tap)
                            dma_start_task(b_task)
                            b_tasks.append(b_task)

                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_await_task(c_tasks[-2])
                            dma_free_task(a_tasks[-2])
                            dma_free_task(b_tasks[-2])

                dma_await_task(c_tasks[-1])

                trace_utils.gen_trace_done_aie2(shim_tile)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
