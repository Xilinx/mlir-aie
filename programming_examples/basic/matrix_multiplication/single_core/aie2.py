#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

import sys
import argparse

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
import aie.utils.trace as trace_utils


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
        "--dtype_in", type=str, choices=["bf16", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out", type=str, choices=["bf16", "i16", "f32", "i32"], default="i32"
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

    dtype_in = None
    if dtype_in_str == "bf16":
        dtype_in = T.bf16
    elif dtype_in_str == "i16":
        dtype_in = T.i16
    dtype_out = None
    if dtype_out_str == "bf16":
        dtype_out = T.bf16
    elif dtype_out_str == "i16":
        dtype_out = T.i16
    elif dtype_out_str == "f32":
        dtype_out = T.f32
    elif dtype_out_str == "i32":
        dtype_out = T.i32

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

        C_sz_in_bytes = C_sz * dtype_out().width // 8

        @device(AIEDevice.npu1_1col)
        def device_body():
            memref_a_ty = T.memref(m, k, dtype_in())
            memref_b_ty = T.memref(k, n, dtype_in())
            memref_c_ty = T.memref(m, n, dtype_out())

            ofifo_memref_a_ty = TypeAttr.get(ObjectFifoType.get(memref_a_ty))
            ofifo_memref_b_ty = TypeAttr.get(ObjectFifoType.get(memref_b_ty))
            ofifo_memref_c_ty = TypeAttr.get(ObjectFifoType.get(memref_c_ty))

            # AIE Core Function declarations
            zero_scalar = external_func(
                f"zero_scalar_{dtype_out_str}", inputs=[memref_c_ty]
            )
            zero = external_func(f"zero_{dtype_out_str}", inputs=[memref_c_ty])
            matmul_scalar = external_func(
                f"matmul_scalar_{dtype_in_str}_{dtype_out_str}",
                inputs=[memref_a_ty, memref_b_ty, memref_c_ty],
            )
            matmul = external_func(
                f"matmul_{dtype_in_str}_{dtype_out_str}",
                inputs=[memref_a_ty, memref_b_ty, memref_c_ty],
            )

            # Tile declarations
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile2_col, compute_tile2_row = 0, 2
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            # Input A
            inA = object_fifo("inA", shim_tile, mem_tile, 2, memref_a_ty)
            memA = object_fifo(
                "memA",
                mem_tile,
                compute_tile2,
                2,
                memref_a_ty,
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
            inB = object_fifo("inB", shim_tile, mem_tile, 2, memref_b_ty)
            memB = object_fifo(
                "memB",
                mem_tile,
                compute_tile2,
                2,
                memref_b_ty,
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
            memC = object_fifo("memC", compute_tile2, mem_tile, 2, memref_c_ty)
            outC = object_fifo(
                "outC",
                mem_tile,
                shim_tile,
                2,
                memref_c_ty,
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
                for _ in for_(0xFFFFFFFF):
                    for _ in for_(tiles) if tiles > 1 else range(1):  # issue #1547
                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        if vectorized:
                            call(zero, [elem_out])
                        else:
                            call(zero_scalar, [elem_out])

                        for _ in (
                            for_(K_div_k) if K_div_k > 1 else range(1)
                        ):  # issue #1547
                            elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                            if vectorized:
                                call(matmul, [elem_in_a, elem_in_b, elem_out])
                            else:
                                call(matmul_scalar, [elem_in_a, elem_in_b, elem_out])
                            memA.release(ObjectFifoPort.Consume, 1)
                            memB.release(ObjectFifoPort.Consume, 1)
                            if K_div_k > 1:
                                yield_([])

                        memC.release(ObjectFifoPort.Produce, 1)
                        if tiles > 1:
                            yield_([])
                    yield_([])

            # To/from AIE-array data movement

            @FuncOp.from_py_func(
                T.memref(A_sz, dtype_in()),
                T.memref(B_sz, dtype_in()),
                T.memref(C_sz, dtype_out()),
            )
            def sequence(A, B, C):

                if enable_tracing:
                    trace_utils.configure_simple_tracing_aie2(
                        compute_tile2,
                        shim_tile,
                        ddr_id=2,
                        size=trace_size,
                        offset=C_sz_in_bytes,
                    )

                # only do 5 tile rows at a time before synchronizing, so we can reuse BDs
                rows_per_block = 5
                for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                    C_row_offset = tile_row_block * rows_per_block * m * N
                    num_tile_rows = min(
                        [rows_per_block, M_div_m - tile_row_block * rows_per_block]
                    )
                    npu_dma_memcpy_nd(
                        metadata="outC",
                        bd_id=0,
                        mem=C,
                        offsets=[0, 0, 0, C_row_offset],
                        sizes=[num_tile_rows, N_div_n, m, n],
                        strides=[m_x_N, n, N, 1],
                    )
                    for tile_row in range(num_tile_rows):
                        A_row_offset = (
                            ((tile_row_block * rows_per_block) + tile_row) * m * K
                        )
                        npu_dma_memcpy_nd(
                            metadata="inA",
                            bd_id=2 * tile_row + 1,
                            mem=A,
                            offsets=[0, 0, 0, A_row_offset],
                            sizes=[N_div_n, K_div_k, m, k],
                            strides=[0, k, K, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata="inB",
                            bd_id=2 * tile_row + 2,
                            mem=B,
                            sizes=[N_div_n, K_div_k, k, n],
                            strides=[n, k_x_N, N, 1],
                        )

                    npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
