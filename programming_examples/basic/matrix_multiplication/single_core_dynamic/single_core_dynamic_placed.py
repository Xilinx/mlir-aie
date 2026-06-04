#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.
#
# Dynamic single-core GEMM design with placed runtime sequence (bf16->f32, 32x32x32 tiles).
#
# Uses the dynamic core body (RTP reads + scf.for loops) for compile-once-run-any-size,
# combined with the placed runtime sequence pattern (shim_dma_single_bd_task / dma_start_task
# / npu_sync) instead of npu_dma_memcpy_nd.  M, K, N are SSA i32 inputs to the
# runtime_sequence so the compiled XCLBIN handles any (multiple-of-tile) shape at runtime
# and --aie-generate-txn-cpp produces a parameterizable C++ function.

import argparse
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith, memref
from aie.helpers.dialects.scf import if_
from aie.extras.dialects.arith import constant
import aie.utils.trace as trace_utils
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype
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
            zero = external_func(
                f"zero_{dtype_out_str}",
                inputs=[c_ty],
                link_with=f"mm_{m}x{k}x{n}.o",
            )
            matmul = external_func(
                f"matmul_{dtype_in_str}_{dtype_out_str}",
                inputs=[a_ty, b_ty, c_ty],
                link_with=f"mm_{m}x{k}x{n}.o",
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

            # Set up a packet-switched flow from core to shim for tracing information
            tiles_to_trace = [compute_tile2]
            if enable_tracing:
                trace_utils.configure_trace(tiles_to_trace)

            @core(compute_tile2, stack_size=0xD00, dynamic_objfifo_lowering=True)
            def core_body():
                c0_idx = constant(0, index=True)
                c1_idx = constant(1, index=True)
                for _ in range_(0xFFFFFFFF):
                    tiles = memref.load(rtp_buf, [c1_idx])
                    K_div_k = memref.load(rtp_buf, [c0_idx])
                    for _ in range_(tiles):
                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_a, elem_in_b, elem_out)
                            memA.release(ObjectFifoPort.Consume, 1)
                            memB.release(ObjectFifoPort.Consume, 1)
                        memC.release(ObjectFifoPort.Produce, 1)

            # Placed runtime_sequence using shim_dma_single_bd_task
            # M, K, N are SSA i32 inputs so the compiled XCLBIN is shape-agnostic.
            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
                T.i32(),  # M
                T.i32(),  # K
                T.i32(),  # N
            )
            def sequence(A, B, C, M, K, N):
                if enable_tracing:
                    trace_utils.start_trace(trace_size=trace_size)

                M_div_m = M // m
                K_div_k = K // k
                N_div_n = N // n
                tiles = M_div_m * N_div_n

                npu_rtp_write("rtp", 0, K_div_k)
                npu_rtp_write("rtp", 1, tiles)

                rows_per_block = 4

                for tile_row_block in range_(ceildiv(M_div_m, rows_per_block)):
                    for pingpong in [0, 1]:
                        C_row_offset = (
                            tile_row_block * rows_per_block * m * N
                            + pingpong * rows_per_block // 2 * m * N
                        )
                        row_base = (
                            tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2
                        )
                        num_tile_rows = arith.minsi(
                            constant(rows_per_block // 2, T.i32()),
                            M_div_m - row_base,
                        )
                        with if_(num_tile_rows > 0, hasElse=False):
                            c_task = shim_dma_single_bd_task(
                                outC,
                                C,
                                offset=C_row_offset,
                                sizes=[num_tile_rows, N_div_n, m, n],
                                strides=[m * N, n, N, 1],
                                transfer_len=N_div_n * m * n,
                                issue_token=True,
                            )
                            dma_start_task(c_task)

                            for tile_row in range(rows_per_block // 2):
                                A_row_offset = (row_base + tile_row) * m * K

                                def emit_ab():
                                    a_task = shim_dma_single_bd_task(
                                        inA,
                                        A,
                                        offset=A_row_offset,
                                        sizes=[N_div_n, K_div_k, m, k],
                                        strides=[0, k, K, 1],
                                        transfer_len=K_div_k * m * k,
                                    )
                                    dma_start_task(a_task)

                                    b_task = shim_dma_single_bd_task(
                                        inB,
                                        B,
                                        sizes=[N_div_n, K_div_k, k, n],
                                        strides=[n, k * N, N, 1],
                                        transfer_len=K_div_k * k * n,
                                    )
                                    dma_start_task(b_task)

                                if tile_row == 0:
                                    emit_ab()
                                else:
                                    with if_(num_tile_rows > tile_row, hasElse=False):
                                        emit_ab()

                            if pingpong > 0:
                                npu_sync(column=0, row=0, direction=0, channel=0)
                            else:
                                with if_(tile_row_block > 0, hasElse=False):
                                    npu_sync(column=0, row=0, direction=0, channel=0)

                npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
