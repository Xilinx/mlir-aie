#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.extras.dialects.ext.scf import _for as range_

dtype_map = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "f32": np.float32,
    "i32": np.int32,
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    args = argparser.parse_args()
    my_matmul(
        args.M, args.K, args.N, args.m, args.k, args.n, args.dtype_in, args.dtype_out
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(M, K, N, m, k, n, dtype_in_str, dtype_out_str):

    assert M % m == 0
    assert K % k == 0
    assert N % n == 0

    if dtype_in_str == "bf16":
        r = 4
        s = 8
        t = 4
    elif dtype_in_str == "i8":
        r = 4
        s = 8
        t = 8
    elif dtype_in_str == "i16":
        r = 4
        s = 4
        t = 4

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = False
    trace_size = 65536

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    A_sz = M * K
    B_sz = K * N
    C_sz = M * N

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Matrix B: KxN, submatrices b: kxn
    k_x_N = k * N

    # Output Matrix C: MxN
    m_x_N = m * N

    with mlir_mod_ctx() as ctx:

        C_sz_in_bytes = C_sz * np.dtype(dtype_out).itemsize // 8

        @device(AIEDevice.npu1_1col)
        def device_body():
            a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
            b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
            c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

            # AIE Core Function declarations
            func_type = "" if vectorized else "scalar_"
            zero = external_func(f"zero_{func_type}{dtype_out_str}", inputs=[c_ty])
            matmul = external_func(
                f"matmul_{func_type}{dtype_in_str}_{dtype_out_str}",
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
                (
                    [
                        (m // r, r * k),
                        (k // s, s),
                        (r, k),
                        (s, 1),
                    ]
                    if vectorized
                    else []
                ),
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
                (
                    [
                        (k // s, s * n),
                        (n // t, t),
                        (s, n),
                        (t, 1),
                    ]
                    if vectorized
                    else []
                ),
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
                (
                    [
                        (m // r, r * n),
                        (r, t),
                        (n // t, r * t),
                        (t, 1),
                    ]
                    if vectorized
                    else []
                ),
            )
            object_fifo_link(memC, outC)

            # Set up a circuit-switched flow from core to shim for tracing information
            if enable_tracing:
                flow(compute_tile2, WireBundle.Trace, 0, shim_tile, WireBundle.DMA, 1)

            # Set up compute tiles

            # Compute tile 2
            @core(compute_tile2, f"mm_{m}x{k}x{n}.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547
                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)

                        for _ in (
                            range_(K_div_k) if K_div_k > 1 else range(1)
                        ):  # issue #1547
                            elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_a, elem_in_b, elem_out)
                            memA.release(ObjectFifoPort.Consume, 1)
                            memB.release(ObjectFifoPort.Consume, 1)

                        memC.release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[dtype_in]],
                np.ndarray[(B_sz,), np.dtype[dtype_in]],
                np.ndarray[(C_sz,), np.dtype[dtype_out]],
            )
            def sequence(A, B, C):

                if enable_tracing:
                    trace_utils.configure_simple_tracing_aie2(
                        compute_tile2,
                        shim_tile,
                        ddr_id=2,
                        size=trace_size,
                        offset=C_sz_in_bytes,
                        events=[
                            PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_0,
                                port_number=1,
                                master=True,
                            ),
                            PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_1,
                                port_number=2,
                                master=True,
                            ),
                            PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_2,
                                port_number=5,
                                master=True,
                            ),
                            trace_utils.CoreEvent.INSTR_EVENT_0,
                            trace_utils.CoreEvent.INSTR_EVENT_1,
                            trace_utils.CoreEvent.MEMORY_STALL,
                            trace_utils.CoreEvent.LOCK_STALL,
                            trace_utils.CoreEvent.INSTR_VECTOR,
                        ],
                    )

                # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
                rows_per_block = 4
                for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                    # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
                    # that's what this loop is for
                    for pingpong in [0, 1]:
                        C_row_offset = (
                            tile_row_block * rows_per_block * m * N
                            + pingpong * rows_per_block // 2 * m * N
                        )
                        row_base = (
                            tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2
                        )
                        bd_id_base = 8 * pingpong
                        num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                        if num_tile_rows <= 0:
                            # At the very last iteration, we may not need a 'pong' iteration
                            break
                        npu_dma_memcpy_nd(
                            metadata=outC,
                            bd_id=bd_id_base,
                            mem=C,
                            offsets=[0, 0, 0, C_row_offset],
                            sizes=[num_tile_rows, N_div_n, m, n],
                            strides=[m_x_N, n, N, 1],
                        )
                        for tile_row in range(num_tile_rows):
                            A_row_offset = (row_base + tile_row) * m * K
                            npu_dma_memcpy_nd(
                                metadata=inA,
                                bd_id=bd_id_base + 2 * tile_row + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_row_offset],
                                sizes=[N_div_n, K_div_k, m, k],
                                strides=[0, k, K, 1],
                            )
                            npu_dma_memcpy_nd(
                                metadata=inB,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=B,
                                sizes=[N_div_n, K_div_k, k, n],
                                strides=[n, k_x_N, N, 1],
                            )
                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_wait(outC)
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
