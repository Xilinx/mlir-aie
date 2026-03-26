#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.
#
# Dynamic single-core GEMM design (bf16->f32, 32x32x32 tiles).
#
# The core processes fixed 32x32x32 tiles. Loop bounds (K iterations and
# tile count) are read at runtime from an RTP buffer, enabling the host
# to run any GEMM shape (multiples of 32) without recompiling the XCLBIN.
#
# The static runtime_sequence in this file is used only for XCLBIN generation
# at the specified M/K/N. A separate hand-written MLIR file provides the
# dynamic runtime_sequence that is compiled to C++ via aie-generate-txn-cpp.
#
# With --dynamic-txn, emits a runtime_sequence that takes M, K, N as SSA
# parameters, allowing the instruction stream itself to be parameterized.

import argparse
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import ForOp, IfOp, WhileOp, condition, yield_
from aie.dialects.arith import (
    constant as arith_constant,
    CmpIOp,
    CmpIPredicate,
    AddIOp,
    SubIOp,
    MulIOp,
    DivUIOp,
    MinSIOp,
    IndexCastOp,
    ExtUIOp,
)
from aie.dialects.memref import LoadOp
from aie.extras.dialects.arith import constant
import aie.utils.trace as trace_utils
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype
from aie.ir import IndexType, IntegerType, InsertionPoint, Block
from aie.extras import types as T


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Dynamic Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a dynamic matrix multiplication design",
    )
    argparser.add_argument("--dev", type=str, choices=["npu2"], default="npu2")
    argparser.add_argument("-M", type=int, default=128)
    argparser.add_argument("-K", type=int, default=128)
    argparser.add_argument("-N", type=int, default=128)
    argparser.add_argument("--dtype_in", type=str, choices=["bf16"], default="bf16")
    argparser.add_argument("--dtype_out", type=str, choices=["f32"], default="f32")
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--dynamic-txn", action="store_true", default=False)
    args = argparser.parse_args()
    my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.dtype_in,
        args.dtype_out,
        args.trace_size,
        args.dynamic_txn,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def _emit_dynamic_sequence(
    A_sz,
    B_sz,
    C_sz,
    dtype_in,
    dtype_out,
    m,
    k,
    n,
    inA,
    inB,
    outC,
    enable_tracing,
    trace_size,
    tiles_to_trace,
    shim_tile,
):
    """Emit a runtime_sequence with M, K, N as SSA i32 parameters.

    This generates the same DMA pattern as the static sequence and the
    hand-written dynamic_gemm_txn.h, but expressed as MLIR SCF ops so that
    M, K, N can vary at runtime without recompiling the XCLBIN.

    The outer tile-row-block loop and inner tile-row loop use scf.for.
    Pingpong (2 half-blocks) is unrolled as a Python loop.

    Note on RTP synchronization: the current core body polls RTP via
    memref.load before consuming ObjectFIFO data. Since the host writes
    RTP before issuing any DMAs, and the core reads RTP before processing,
    the DMA start acts as an implicit ordering barrier. Lock-based
    synchronization should be added for robustness if the design is used
    with overlapping invocations.
    """
    rows_per_block = 4

    @runtime_sequence(
        np.ndarray[(A_sz,), np.dtype[dtype_in]],
        np.ndarray[(B_sz,), np.dtype[dtype_in]],
        np.ndarray[(C_sz,), np.dtype[dtype_out]],
        T.i32(),  # M
        T.i32(),  # K
        T.i32(),  # N
    )
    def sequence(A, B, C, M_param, K_param, N_param):
        i32_ty = IntegerType.get_signless(32)
        i64_ty = IntegerType.get_signless(64)
        idx_ty = IndexType.get()

        # Helper: cast i32 SSA value to i64 (required by npu_dma_memcpy_nd)
        def to_i64(val):
            return ExtUIOp(i64_ty, val).result

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

        # Static tile-size constants
        m_const = arith_constant(i32_ty, m)
        k_const = arith_constant(i32_ty, k)
        n_const = arith_constant(i32_ty, n)
        c0 = arith_constant(i32_ty, 0)
        c1 = arith_constant(i32_ty, 1)
        c2 = arith_constant(i32_ty, 2)
        rpb_const = arith_constant(i32_ty, rows_per_block)
        rpb_half_const = arith_constant(i32_ty, rows_per_block // 2)

        # Derived SSA values
        # M_div_m = M / m
        M_div_m = DivUIOp(M_param, m_const).result
        # K_div_k = K / k
        K_div_k = DivUIOp(K_param, k_const).result
        # N_div_n = N / n
        N_div_n = DivUIOp(N_param, n_const).result
        # tiles = M_div_m * N_div_n
        tiles = MulIOp(M_div_m, N_div_n).result

        # Write RTP values via symbolic buffer reference
        npu_rtp_write("rtp", 0, K_div_k)
        npu_rtp_write("rtp", 1, tiles)

        # ceildiv(M_div_m, rows_per_block) as SSA:
        #   (M_div_m + rows_per_block - 1) / rows_per_block
        rpb_minus1 = arith_constant(i32_ty, rows_per_block - 1)
        M_div_m_plus = AddIOp(M_div_m, rpb_minus1).result
        tile_row_blocks = DivUIOp(M_div_m_plus, rpb_const).result

        # Convert loop bounds to index type for scf.for
        c0_idx = IndexCastOp(idx_ty, c0).result
        c1_idx = IndexCastOp(idx_ty, c1).result
        tile_row_blocks_idx = IndexCastOp(idx_ty, tile_row_blocks).result

        # Outer loop: tile_row_block = 0 .. ceildiv(M_div_m, rows_per_block)
        # Carries a "first_batch" flag (i32: 1=first, 0=not first) to
        # know when to skip the initial dma_wait sync.
        first_batch_init = c1  # 1 = true (first batch)
        outer_for = ForOp(c0_idx, tile_row_blocks_idx, c1_idx, [first_batch_init])
        with InsertionPoint(outer_for.body):
            trb_iv = outer_for.induction_variable  # index
            first_batch_arg = outer_for.inner_iter_args[0]  # i32

            # Convert induction variable to i32 for arithmetic
            trb_i32 = IndexCastOp(i32_ty, trb_iv).result

            # Unroll pingpong = 0, 1
            current_first_batch = first_batch_arg
            for pingpong in [0, 1]:
                pingpong_const = arith_constant(i32_ty, pingpong)
                pp_half = arith_constant(i32_ty, pingpong * (rows_per_block // 2))
                bd_id_base = 8 * pingpong

                # row_base = trb * rows_per_block + pingpong * (rpb/2)
                row_base = AddIOp(MulIOp(trb_i32, rpb_const).result, pp_half).result

                # num_tile_rows = min(rpb/2, M_div_m - row_base)
                remaining = SubIOp(M_div_m, row_base).result
                num_tile_rows = MinSIOp(rpb_half_const, remaining).result

                # Guard: skip this pingpong half if num_tile_rows <= 0
                has_rows = CmpIOp(CmpIPredicate.sgt, num_tile_rows, c0).result
                guard_if = IfOp(has_rows, results_=[i32_ty], has_else=True)

                with InsertionPoint(guard_if.then_block):
                    # C_row_offset = row_base * m * N
                    C_row_offset = MulIOp(
                        MulIOp(row_base, m_const).result, N_param
                    ).result

                    # C output BD
                    npu_dma_memcpy_nd(
                        metadata=outC,
                        bd_id=bd_id_base,
                        mem=C,
                        offsets=[0, 0, 0, to_i64(C_row_offset)],
                        sizes=[to_i64(num_tile_rows), to_i64(N_div_n), m, n],
                        strides=[
                            to_i64(MulIOp(m_const, N_param).result),
                            n,
                            to_i64(N_param),
                            1,
                        ],
                    )

                    # Unroll tile_row = 0, 1 (max rows_per_block//2 = 2).
                    # bd_id must be a static integer, so we cannot use scf.for.
                    # tile_row=0 is always valid (has_rows guard above).
                    # tile_row=1 is guarded by num_tile_rows > 1.
                    for tile_row in range(rows_per_block // 2):
                        a_bd_id = bd_id_base + 2 * tile_row + 1
                        b_bd_id = bd_id_base + 2 * tile_row + 2

                        def _emit_ab_bds(_row_base, _tile_row, _a_bd, _b_bd):
                            tr_const = arith_constant(i32_ty, _tile_row)
                            abs_row = AddIOp(_row_base, tr_const).result
                            A_row_offset = MulIOp(
                                MulIOp(abs_row, m_const).result, K_param
                            ).result
                            npu_dma_memcpy_nd(
                                metadata=inA,
                                bd_id=_a_bd,
                                mem=A,
                                offsets=[0, 0, 0, to_i64(A_row_offset)],
                                sizes=[to_i64(N_div_n), to_i64(K_div_k), m, k],
                                strides=[0, k, to_i64(K_param), 1],
                            )
                            npu_dma_memcpy_nd(
                                metadata=inB,
                                bd_id=_b_bd,
                                mem=B,
                                sizes=[to_i64(N_div_n), to_i64(K_div_k), k, n],
                                strides=[
                                    n,
                                    to_i64(MulIOp(k_const, N_param).result),
                                    to_i64(N_param),
                                    1,
                                ],
                            )

                        if tile_row == 0:
                            # Always valid when has_rows
                            _emit_ab_bds(row_base, 0, a_bd_id, b_bd_id)
                        else:
                            # Guard: only emit if num_tile_rows > tile_row
                            has_more = CmpIOp(
                                CmpIPredicate.sgt,
                                num_tile_rows,
                                arith_constant(i32_ty, tile_row),
                            ).result
                            row_if = IfOp(has_more, has_else=False)
                            with InsertionPoint(row_if.then_block):
                                _emit_ab_bds(row_base, tile_row, a_bd_id, b_bd_id)
                                yield_([])

                    # Wait for previous batch to complete before reusing BDs.
                    # Use npu_sync directly (instead of dma_wait inside scf.if)
                    # to avoid block terminator issues with partial conversion.
                    not_first = CmpIOp(CmpIPredicate.eq, current_first_batch, c0).result
                    sync_if = IfOp(not_first, has_else=False)
                    with InsertionPoint(sync_if.then_block):
                        npu_sync(column=0, row=0, direction=0, channel=0)
                        yield_([])

                    # After this half-block, no longer first batch
                    yield_([c0])
                # End guard_if then

                with InsertionPoint(guard_if.else_block):
                    # No rows to process, pass through first_batch unchanged
                    yield_([current_first_batch])

                current_first_batch = guard_if.results[0]
            # End pingpong unroll

            yield_([current_first_batch])
        # End outer_for

        # Final sync: wait for the last batch
        npu_sync(column=0, row=0, direction=0, channel=0)


def my_matmul(dev, M, K, N, dtype_in_str, dtype_out_str, trace_size, dynamic_txn=False):
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

    vectorized = True
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
            )

            # Set up tracing
            tiles_to_trace = [compute_tile2]
            if enable_tracing:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            # Core body with dynamic loop bounds via RTP
            @core(
                compute_tile2,
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

            if dynamic_txn:
                _emit_dynamic_sequence(
                    A_sz,
                    B_sz,
                    C_sz,
                    dtype_in,
                    dtype_out,
                    m,
                    k,
                    n,
                    inA,
                    inB,
                    outC,
                    enable_tracing,
                    trace_size,
                    tiles_to_trace,
                    shim_tile,
                )
            else:
                # Static runtime_sequence for XCLBIN generation
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

                    # Simple DMA sequence for the static case
                    # Issue all tile rows at once (simplified from single_core.py)
                    rows_per_block = 4
                    for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
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
                            num_tile_rows = min(
                                [rows_per_block // 2, M_div_m - row_base]
                            )
                            if num_tile_rows <= 0:
                                break
                            npu_dma_memcpy_nd(
                                metadata=outC,
                                bd_id=bd_id_base,
                                mem=C,
                                offsets=[0, 0, 0, C_row_offset],
                                sizes=[num_tile_rows, N_div_n, m, n],
                                strides=[m * N, n, N, 1],
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
                                    strides=[n, k * N, N, 1],
                                )
                            if tile_row_block > 0 or (
                                tile_row_block == 0 and pingpong > 0
                            ):
                                dma_wait(outC)
                        dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
