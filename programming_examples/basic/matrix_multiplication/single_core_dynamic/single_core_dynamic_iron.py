#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.
#
# IRON-level dynamic single-core GEMM (bf16->f32, 32x32x32 tiles).
#
# Uses IRON high-level abstractions (Worker, ObjectFifo, Kernel, Runtime,
# Program) with RTP-based dynamic loop bounds for compile-once-run-any-size.
# Fixed tile sizes (32x32x32) are compiled once; the host sets loop bounds
# (K iterations, tile count) via RTP before each run.

import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer, str_to_dtype
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D

from aie.dialects.scf import WhileOp, condition, yield_
from aie.dialects.arith import CmpIOp, CmpIPredicate, AddIOp, IndexCastOp
from aie.dialects.memref import LoadOp
from aie.extras.dialects.arith import constant
from aie.ir import IndexType, InsertionPoint
from aie.extras import types as T


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Dynamic Matrix Multiplication MLIR Design (Single Core, IRON)",
        description="Emits MLIR code for a dynamic matrix multiplication design using IRON APIs",
    )
    argparser.add_argument("--dev", type=str, choices=["npu2"], default="npu2")
    argparser.add_argument("-M", type=int, default=128)
    argparser.add_argument("-K", type=int, default=128)
    argparser.add_argument("-N", type=int, default=128)
    argparser.add_argument("--dtype_in", type=str, choices=["bf16"], default="bf16")
    argparser.add_argument("--dtype_out", type=str, choices=["f32"], default="f32")
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    module = my_matmul(
        args.dev, args.M, args.K, args.N, args.dtype_in, args.dtype_out, args.trace_size
    )
    print(module)


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

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Define tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # AIE Core Function declarations
    zero_kernel = Kernel(f"zero_{dtype_out_str}", f"mm_{m}x{k}x{n}.o", [c_ty])
    matmul_kernel = Kernel(
        f"matmul_{dtype_in_str}_{dtype_out_str}",
        f"mm_{m}x{k}x{n}.o",
        [a_ty, b_ty, c_ty],
    )

    # AIE-array data movement with object fifos
    # Input A
    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # Input B
    inB = ObjectFifo(b_ty, name="inB")
    b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    # Output C
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    # RTP buffer: [0] = K_div_k, [1] = tiles (M_div_m * N_div_n)
    rtp_buf = Buffer(
        type=np.ndarray[(16,), np.dtype[np.int32]],
        name="rtp",
        use_write_rtp=True,
    )

    # Core function with dynamic loop bounds via RTP
    def core_fn(of_a, of_b, of_c, zero, matmul, rtp):
        idx_ty = IndexType.get()
        c0_idx = constant(0, index=True)
        c1_idx = constant(1, index=True)

        # Read tile count from RTP[1]
        tiles_i32 = LoadOp(rtp.op, [c1_idx]).result
        tiles_idx = IndexCastOp(idx_ty, tiles_i32).result

        # Tile loop (scf.while for dynamic bounds)
        tile_while = WhileOp([idx_ty], [c0_idx])

        # "before" region: check condition
        before_block = tile_while.before.blocks.append(idx_ty)
        with InsertionPoint(before_block):
            tile_iter = before_block.arguments[0]
            tile_cond = CmpIOp(CmpIPredicate.slt, tile_iter, tiles_idx).result
            condition(tile_cond, [tile_iter])

        # "after" region: loop body
        after_block = tile_while.after.blocks.append(idx_ty)
        with InsertionPoint(after_block):
            tile_iter = after_block.arguments[0]

            elem_out = of_c.acquire(1)
            zero(elem_out)

            # Read K_div_k from RTP[0]
            k_iters_i32 = LoadOp(rtp.op, [c0_idx]).result
            k_iters_idx = IndexCastOp(idx_ty, k_iters_i32).result

            # K accumulation loop (scf.while)
            k_while = WhileOp([idx_ty], [c0_idx])

            k_before = k_while.before.blocks.append(idx_ty)
            with InsertionPoint(k_before):
                k_iter = k_before.arguments[0]
                k_cond = CmpIOp(CmpIPredicate.slt, k_iter, k_iters_idx).result
                condition(k_cond, [k_iter])

            k_after = k_while.after.blocks.append(idx_ty)
            with InsertionPoint(k_after):
                k_iter = k_after.arguments[0]

                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)

                k_next = AddIOp(k_iter, c1_idx).result
                yield_([k_next])

            of_c.release(1)

            tile_next = AddIOp(tile_iter, c1_idx).result
            yield_([tile_next])

    # Create worker with dynamic objfifo lowering
    worker = Worker(
        core_fn,
        [
            memA.cons(),
            memB.cons(),
            memC.prod(),
            zero_kernel,
            matmul_kernel,
            rtp_buf,
        ],
        stack_size=0xD00,
        dynamic_objfifo_lowering=True,
    )

    # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
    rows_per_block = 4

    # Define tensor access patterns for inputs/outputs
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=N_div_n, prune_step=False
    )
    b_tap = TensorTiler2D.group_tiler(
        (K, N),
        (k, n),
        (K_div_k, N_div_n),
        tile_group_col_major=True,
        prune_step=False,
    )[0]
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m, n), (rows_per_block // 2, N_div_n), prune_step=False
    )
    c_index = 0

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(worker)

        # Write RTP values for the static compilation size
        rt.write_rtp(rtp_buf, 0, K_div_k)
        rt.write_rtp(rtp_buf, 1, tiles)

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

    # Create the program from the device type and runtime
    dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components and generate MLIR module
    module = my_program.resolve_program(SequentialPlacer())
    return module


if __name__ == "__main__":
    main()
