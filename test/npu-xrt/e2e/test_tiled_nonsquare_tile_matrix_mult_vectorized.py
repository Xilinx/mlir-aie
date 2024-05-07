# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.


from __future__ import annotations

from pathlib import Path
import sys

from aie.compiler.util import (
    compile_with_vectorization,
    make_xclbin,
)
from aie.dialects import aie, aievec, aiex, builtin, pdl, vector
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)
from aie.dialects.linalg.opdsl.ops.core_named_ops import fill as linalg_fill
from aie.dialects.scf import for_ as range_, yield_
from aie.dialects.transform import any_op_t, apply_registered_pass, get_parent_op
from aie.dialects.transform.extras import named_sequence
from aie.dialects.transform.loop import loop_unroll
from aie.dialects.transform.structured import structured_match
from aie.extras.context import ExplicitlyManagedModule
from aie.extras.dialects.ext import arith, func, linalg
from aie.extras.runtime.passes import Pipeline, run_pipeline

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
import aie.extras.types as T
from aie.extras.util import find_ops
from aie.ir import AffineDimExpr, AffineMap, StringAttr, UnitAttr
from aie.util import tiling_calculator_n_tiles
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np
import pytest

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


DMA = WireBundle.DMA
South = WireBundle.South
North = WireBundle.North
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release

M = N = 32

tile_rows_A, tile_cols_A = 2, 1
tile_rows_B, tile_cols_B = 1, 2
tile_rows_C, tile_cols_C = 2, 2

tile_m_A, tile_n_A = M // tile_rows_A, N // tile_cols_A
tile_m_B, tile_n_B = M // tile_rows_B, N // tile_cols_B
tile_m_C, tile_n_C = M // tile_rows_C, N // tile_cols_C


@func.func(sym_visibility="private")
def matmul_i32_i32(
    A: T.memref(tile_m_A, tile_n_A, T.i32()),
    B: T.memref(tile_m_B, tile_n_B, T.i32()),
    C: T.memref(tile_m_C, tile_n_C, T.i32()),
):
    linalg.matmul(A, B, C)


def test_tiled_nonsquare_tile_matrix_mult_vectorized(ctx: MLIRContext, workdir: Path):
    (
        _,
        _,
        (d1_size_A, d1_stride_A),
        (d0_size_A, d0_stride_A),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_A, n_tile_cols=tile_cols_A
    )
    (
        _,
        _,
        (d1_size_B, d1_stride_B),
        (d0_size_B, d0_stride_B),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_B, n_tile_cols=tile_cols_B
    )
    (
        _,
        _,
        (d1_size_C, d1_stride_C),
        (d0_size_C, d0_stride_C),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_C, n_tile_cols=tile_cols_C
    )

    npu_insts = aiex.npu.get_prolog()

    mod_aie = ExplicitlyManagedModule()

    @aie.device(AIEDevice.npu)
    def npu():
        matmul_i32_i32.emit(decl=True)
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(tile_0_2, (tile_m_A, tile_n_A), T.i32())
        buffer_0_2_b = aie.buffer(tile_0_2, (tile_m_B, tile_n_B), T.i32())
        # out
        buffer_0_2_c = aie.buffer(tile_0_2, (tile_m_C, tile_n_C), T.i32())

        # input
        lock_0_1_read_in_a = aie.lock(tile_0_1, lock_id=0, init=1)
        lock_0_1_write_out_a = aie.lock(tile_0_1, lock_id=1, init=0)
        lock_0_1_read_in_b = aie.lock(tile_0_1, lock_id=2, init=1)
        lock_0_1_write_out_b = aie.lock(tile_0_1, lock_id=3, init=0)
        # output/returning
        lock_0_1_read_in_c = aie.lock(tile_0_1, lock_id=4, init=1)
        lock_0_1_write_out_c = aie.lock(tile_0_1, lock_id=5, init=0)

        lock_0_2_read_in_a = aie.lock(tile_0_2, lock_id=0, init=1)
        lock_0_2_use_a = aie.lock(tile_0_2, lock_id=1, init=0)
        lock_0_2_read_in_b = aie.lock(tile_0_2, lock_id=2, init=1)
        lock_0_2_use_b = aie.lock(tile_0_2, lock_id=3, init=0)
        lock_0_2_use_c = aie.lock(tile_0_2, lock_id=4, init=1)
        lock_0_2_write_out_c = aie.lock(tile_0_2, lock_id=5, init=0)

        # input flow
        # a
        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        # b
        aie.flow(tile_0_0, DMA, 1, tile_0_1, DMA, 1)
        aie.flow(tile_0_1, DMA, 1, tile_0_2, DMA, 1)
        # output flow
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        col = 0
        # in A
        channel_index = 0
        ddr_id = 0
        offsets = [
            0,
            # A tiles are "fat" so need to offset by rows (i.e. d1 dim)
            0 + d1_size_A * d1_stride_A,
        ]
        for i, bd_id in enumerate(range(2)):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_A * tile_n_A,
                    buffer_offset=offsets[i],
                    ddr_id=ddr_id,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(MM2S, channel_index, col, bd_id)
            )

        # in B
        channel_index = 1
        ddr_id = 1
        for bd_id in range(bd_id + 1, bd_id + 1 + 4, 2):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    buffer_offset=0,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(MM2S, channel_index, col, bd_id)
            )
            bd_id += 1
            # B tiles are "tall" so need to offset by cols (i.e. d0 dim)
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    buffer_offset=d0_size_B * d0_stride_B,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(MM2S, channel_index, col, bd_id)
            )

        # out C
        channel_index = 0
        ddr_id = 2
        offsets = [
            0,
            0 + d0_size_C * d0_stride_C,
            d1_size_C * d1_stride_C,
            d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
        ]

        for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + 4)):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_C * tile_n_C,
                    buffer_offset=offsets[i],
                    ddr_id=ddr_id,
                    d1_size=d1_size_C,
                    d1_stride=d1_stride_C,
                    d0_size=d0_size_C,
                    d0_stride=d0_stride_C,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            npu_insts.extend(
                aiex.npu.sync(
                    channel=0,
                    column=0,
                    column_num=1,
                    direction=0,
                    row=0,
                    row_num=1,
                )
            )

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1_a = aie.buffer(tile_0_1, (tile_m_A, tile_n_A), T.i32())
            buffer_0_1_b = aie.buffer(tile_0_1, (tile_m_B, tile_n_B), T.i32())
            # output flow
            buffer_0_1_c = aie.buffer(tile_0_1, (tile_m_C, tile_n_C), T.i32())

            @aie.dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_1_read_in_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_a)
                aie.use_lock(lock_0_1_write_out_a, Release)

            @aie.dma(MM2S, 0, num_blocks=2)
            def dma2():
                aie.use_lock(lock_0_1_write_out_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_a)
                aie.use_lock(lock_0_1_write_out_a, Release)

            @aie.another_bd(dma2)
            def dma2point5():
                aie.use_lock(lock_0_1_write_out_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_a)
                aie.use_lock(lock_0_1_read_in_a, Release)

            @aie.dma(S2MM, 1)
            def dma3():
                aie.use_lock(lock_0_1_read_in_b, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_b)
                aie.use_lock(lock_0_1_write_out_b, Release)

            @aie.dma(MM2S, 1)
            def dma4():
                aie.use_lock(lock_0_1_write_out_b, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_b)
                aie.use_lock(lock_0_1_read_in_b, Release)

            @aie.dma(S2MM, 2)
            def dma5():
                aie.use_lock(lock_0_1_read_in_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_write_out_c, Release)

            @aie.dma(MM2S, 2)
            def dma6():
                aie.use_lock(lock_0_1_write_out_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_read_in_c, Release)

            aie.end()

        @aie.mem(tile_0_2)
        def mem_0_2():
            # input
            @aie.dma(S2MM, 0)
            def dma1():
                aie.use_lock(lock_0_2_read_in_a, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_a)
                aie.use_lock(lock_0_2_use_a, Release)

            @aie.dma(S2MM, 1)
            def dma2():
                aie.use_lock(lock_0_2_read_in_b, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_b)
                aie.use_lock(lock_0_2_use_b, Release)

            # output
            @aie.dma(MM2S, 0)
            def dma3():
                aie.use_lock(lock_0_2_write_out_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_2_c)
                aie.use_lock(lock_0_2_use_c, Release)

            aie.end()

        @aie.core(tile_0_2)
        def core():
            for _ in range_(0, tile_rows_C):
                for _ in range_(0, tile_cols_C):
                    # wait on both in and out to be ready
                    # these have to be acge for some reason...
                    aie.use_lock(lock_0_2_use_a, AcquireGreaterEqual)
                    aie.use_lock(lock_0_2_use_b, AcquireGreaterEqual)
                    aie.use_lock(lock_0_2_use_c, AcquireGreaterEqual)

                    linalg_fill(arith.constant(0), outs=[buffer_0_2_c])
                    matmul_i32_i32(buffer_0_2_a, buffer_0_2_b, buffer_0_2_c)

                    aie.use_lock(lock_0_2_read_in_a, Release)
                    aie.use_lock(lock_0_2_read_in_b, Release)
                    aie.use_lock(lock_0_2_write_out_c, Release)
                    yield_([])
                yield_([])

    mod_aie.finish()
    mod_aievec = ExplicitlyManagedModule()

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul_i32_i32.emit(force=True)

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence("affine_unroll", [any_op_t()], [])
        def affine_unroll(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            new_func = apply_registered_pass(
                any_op_t(), func, "convert-linalg-to-affine-loops"
            )
            m = structured_match(any_op_t(), new_func, ops=["arith.addi"])
            loop = get_parent_op(pdl.op_t(), m, op_name="affine.for")
            # unroll inner loop
            loop_unroll(loop, 8)

        @named_sequence("affine_super_vectorize", [any_op_t()], [])
        def super_vectorize(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            func = apply_registered_pass(
                any_op_t(),
                func,
                "affine-super-vectorize",
                options="virtual-vector-size=16",
            )
            func = apply_registered_pass(any_op_t(), func, "canonicalize")
            mod = apply_registered_pass(
                any_op_t(),
                target,
                "convert-vector-to-aievec",
                options="aie-target=aieml",
            )

    mod_aievec.finish()

    affine_loops = run_pipeline(
        mod_aievec,
        Pipeline()
        .transform_interpreter(
            entry_point="affine_unroll",
            debug_payload_root_tag="payload",
        )
        .canonicalize()
        .cse(),
    )

    super_vec = run_pipeline(
        affine_loops,
        Pipeline()
        .transform_interpreter(
            entry_point="affine_super_vectorize",
            debug_payload_root_tag="payload",
        )
        .lower_affine(),
    )

    mod_aievec = find_ops(
        super_vec.operation,
        lambda x: "transform.target_tag" in x.attributes,
        single=True,
    )

    compile_with_vectorization(mod_aie, mod_aievec, workdir)
    xclbin_path = make_xclbin(mod_aie, workdir)
    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_npu_instructions(npu_insts)
        views = xclbin.mmap_buffers([(M, N), (M, N), (M, N)], np.int32)

        wrap_A = np.asarray(views[0])
        wrap_B = np.asarray(views[1])
        wrap_C = np.asarray(views[2])

        A = np.random.randint(0, 10, (M, N), dtype=np.int32)
        B = np.random.randint(0, 10, (M, N), dtype=np.int32)
        C = np.zeros((M, N), dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)
                assert False


def test_tiled_nonsquare_tile_matrix_mult_vectorized_sugar(
    ctx: MLIRContext, workdir: Path
):
    (
        _,
        _,
        (d1_size_A, d1_stride_A),
        (d0_size_A, d0_stride_A),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_A, n_tile_cols=tile_cols_A
    )
    (
        _,
        _,
        (d1_size_B, d1_stride_B),
        (d0_size_B, d0_stride_B),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_B, n_tile_cols=tile_cols_B
    )
    (
        _,
        _,
        (d1_size_C, d1_stride_C),
        (d0_size_C, d0_stride_C),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_C, n_tile_cols=tile_cols_C
    )

    npu_insts = aiex.npu.get_prolog()

    mod_aie = ExplicitlyManagedModule()

    @aie.device(AIEDevice.npu)
    def npu():
        matmul_i32_i32.emit(decl=True)
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(tile_0_2, (tile_m_A, tile_n_A), T.i32())
        buffer_0_2_b = aie.buffer(tile_0_2, (tile_m_B, tile_n_B), T.i32())
        # out
        buffer_0_2_c = aie.buffer(tile_0_2, (tile_m_C, tile_n_C), T.i32())

        # input
        lock_0_1_read_in_a = aie.lock(tile_0_1, lock_id=0, init=1)
        lock_0_1_write_out_a = aie.lock(tile_0_1, lock_id=1, init=0)

        lock_0_2_read_in_a = aie.lock(tile_0_2, lock_id=0, init=1)
        lock_0_2_use_a = aie.lock(tile_0_2, lock_id=1, init=0)
        lock_0_2_read_in_b = aie.lock(tile_0_2, lock_id=2, init=1)
        lock_0_2_use_b = aie.lock(tile_0_2, lock_id=3, init=0)
        lock_0_2_use_c = aie.lock(tile_0_2, lock_id=4, init=1)
        lock_0_2_write_out_c = aie.lock(tile_0_2, lock_id=5, init=0)

        # input flow
        # a
        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        # b
        aie.flow(tile_0_0, DMA, 1, tile_0_1, DMA, 1)
        aie.flow(tile_0_1, DMA, 1, tile_0_2, DMA, 1)
        # output flow
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        col = 0
        # in A
        channel_index = 0
        ddr_id = 0
        offsets = [
            0,
            # A tiles are "fat" so need to offset by rows (i.e. d1 dim)
            0 + d1_size_A * d1_stride_A,
        ]
        for i, bd_id in enumerate(range(2)):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_A * tile_n_A,
                    buffer_offset=offsets[i],
                    ddr_id=ddr_id,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(MM2S, channel_index, col, bd_id)
            )

        # in B
        channel_index = 1
        col = 0
        ddr_id = 1
        for bd_id in range(bd_id + 1, bd_id + 1 + 4, 2):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    buffer_offset=0,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(MM2S, channel_index, col, bd_id)
            )
            bd_id += 1
            # B tiles are "tall" so need to offset by cols (i.e. d0 dim)
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    buffer_offset=d0_size_B * d0_stride_B,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(MM2S, channel_index, col, bd_id)
            )

        # out C
        channel_index = 0
        col = 0
        ddr_id = 2
        offsets = [
            0,
            0 + d0_size_C * d0_stride_C,
            d1_size_C * d1_stride_C,
            d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
        ]

        for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + 4)):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_C * tile_n_C,
                    buffer_offset=offsets[i],
                    ddr_id=ddr_id,
                    d1_size=d1_size_C,
                    d1_stride=d1_stride_C,
                    d0_size=d0_size_C,
                    d0_stride=d0_stride_C,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            npu_insts.extend(
                aiex.npu.sync(
                    channel=0,
                    column=0,
                    column_num=1,
                    direction=0,
                    row=0,
                    row_num=1,
                )
            )

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1_a = aie.buffer(tile_0_1, (tile_m_A, tile_n_A), T.i32())
            buffer_0_1_b = aie.buffer(tile_0_1, (tile_m_B, tile_n_B), T.i32())
            # output flow
            buffer_0_1_c = aie.buffer(tile_0_1, (tile_m_C, tile_n_C), T.i32())

            @aie.dma(S2MM, 0)
            def dma1():
                aiex.process_bd(lock_0_1_read_in_a, buffer_0_1_a, lock_0_1_write_out_a)

            @aie.dma(MM2S, 0, num_blocks=2)
            def dma2():
                aiex.process_bd(
                    lock_0_1_write_out_a, buffer_0_1_a, lock_0_1_write_out_a
                )

            @aie.another_bd(dma2)
            def dma2point5():
                aiex.process_bd(lock_0_1_write_out_a, buffer_0_1_a, lock_0_1_read_in_a)

            aiex.forward_bd(tile_0_1, buffer_0_1_b, 1)
            aiex.forward_bd(tile_0_1, buffer_0_1_c, 2)

            aie.end()

        @aie.mem(tile_0_2)
        def mem_0_2():
            # input
            @aie.dma(S2MM, 0)
            def dma1():
                aiex.process_bd(lock_0_2_read_in_a, buffer_0_2_a, lock_0_2_use_a)

            @aie.dma(S2MM, 1)
            def dma2():
                aiex.process_bd(lock_0_2_read_in_b, buffer_0_2_b, lock_0_2_use_b)

            # output
            @aie.dma(MM2S, 0)
            def dma3():
                aiex.process_bd(lock_0_2_write_out_c, buffer_0_2_c, lock_0_2_use_c)

            aie.end()

        @aie.core(tile_0_2)
        def core():
            for _ in range_(0, tile_rows_C):
                for _ in range_(0, tile_cols_C):
                    with (
                        aiex.hold_lock(lock_0_2_use_a, lock_0_2_read_in_a),
                        aiex.hold_lock(lock_0_2_use_b, lock_0_2_read_in_b),
                        aiex.hold_lock(lock_0_2_use_c, lock_0_2_write_out_c),
                    ):
                        linalg_fill(arith.constant(0), outs=[buffer_0_2_c])
                        matmul_i32_i32(buffer_0_2_a, buffer_0_2_b, buffer_0_2_c)

                    yield_([])
                yield_([])

    mod_aie.finish()
    mod_aievec = ExplicitlyManagedModule()

    @builtin.module(attrs={"transform.target_tag": StringAttr.get("payload")})
    def payload():
        matmul_i32_i32.emit(force=True)

    @builtin.module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod_transform():
        @named_sequence("affine_unroll", [any_op_t()], [])
        def affine_unroll(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            new_func = apply_registered_pass(
                any_op_t(), func, "convert-linalg-to-affine-loops"
            )
            m = structured_match(any_op_t(), new_func, ops=["arith.addi"])
            loop = get_parent_op(pdl.op_t(), m, op_name="affine.for")
            # unroll inner loop
            loop_unroll(loop, 8)

        @named_sequence("affine_super_vectorize", [any_op_t()], [])
        def super_vectorize(target: any_op_t()):
            func = structured_match(any_op_t(), target, ops=["func.func"])
            func = apply_registered_pass(
                any_op_t(),
                func,
                "affine-super-vectorize",
                options="virtual-vector-size=16",
            )
            func = apply_registered_pass(any_op_t(), func, "canonicalize")
            mod = apply_registered_pass(
                any_op_t(),
                target,
                "convert-vector-to-aievec",
                options="aie-target=aieml",
            )

    mod_aievec.finish()

    affine_loops = run_pipeline(
        mod_aievec,
        Pipeline()
        .transform_interpreter(
            entry_point="affine_unroll",
            debug_payload_root_tag="payload",
        )
        .canonicalize()
        .cse(),
    )

    super_vec = run_pipeline(
        affine_loops,
        Pipeline()
        .transform_interpreter(
            entry_point="affine_super_vectorize",
            debug_payload_root_tag="payload",
        )
        .lower_affine(),
    )
    print(super_vec)

    mod_aievec = find_ops(
        super_vec.operation,
        lambda x: "transform.target_tag" in x.attributes,
        single=True,
    )

    compile_with_vectorization(mod_aie, mod_aievec, workdir)
    xclbin_path = make_xclbin(mod_aie, workdir)
    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_npu_instructions(npu_insts)
        views = xclbin.mmap_buffers([(M, N), (M, N), (M, N)], np.int32)

        wrap_A = np.asarray(views[0])
        wrap_B = np.asarray(views[1])
        wrap_C = np.asarray(views[2])

        A = np.random.randint(0, 10, (M, N), dtype=np.int32)
        B = np.random.randint(0, 10, (M, N), dtype=np.int32)
        C = np.zeros((M, N), dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)
                assert False


@func.func(sym_visibility="private")
def matmul_i32_i32_already_vectorized(
    A: T.memref(tile_m_A, tile_n_A, T.i32()),
    B: T.memref(tile_m_B, tile_n_B, T.i32()),
    C: T.memref(tile_m_C, tile_n_C, T.i32()),
):
    vec16int32 = T.vector(16, T.i32())
    vec16int64 = T.vector(16, T.i64())
    perm_map = AffineMap.get(2, 0, [AffineDimExpr.get(1)])

    c0 = arith.constant(0, index=True)
    for j in range_(0, 16):
        c_vec = aievec.upd(vec16int32, C, [j, c0])
        accum = aievec.ups(vec16int64, c_vec)
        for k in range_(0, 32, 8):
            a_vec = aievec.upd(vec16int32, A, [j, k])
            for i in range(0, 8):
                broad_a = aievec.broadcast(vec16int32, a_vec, idx=i)
                b_vec = aievec.upd(vec16int32, B, [k + i, c0])
                accum = aievec.mac_elem(vec16int64, broad_a, b_vec, accum)

            shift_round_sat = aievec.srs(vec16int32, accum, arith.constant(0))
            vector.transfer_write(
                None,
                shift_round_sat,
                C,
                [j, c0],
                permutation_map=perm_map,
                in_bounds=[True],
            )
            yield_([])
        yield_([])


def test_tiled_nonsquare_tile_matrix_mult_vectorized_sugar_already_vectorized(
    ctx: MLIRContext, workdir: Path
):
    (
        _,
        _,
        (d1_size_A, d1_stride_A),
        (d0_size_A, d0_stride_A),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_A, n_tile_cols=tile_cols_A
    )
    (
        _,
        _,
        (d1_size_B, d1_stride_B),
        (d0_size_B, d0_stride_B),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_B, n_tile_cols=tile_cols_B
    )
    (
        _,
        _,
        (d1_size_C, d1_stride_C),
        (d0_size_C, d0_stride_C),
    ) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=tile_rows_C, n_tile_cols=tile_cols_C
    )

    npu_insts = aiex.npu.get_prolog()

    mod_aie = ExplicitlyManagedModule()

    @aie.device(AIEDevice.npu)
    def npu():
        matmul_i32_i32_already_vectorized.emit(decl=True)
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        # in
        buffer_0_2_a = aie.buffer(tile_0_2, (tile_m_A, tile_n_A), T.i32())
        buffer_0_2_b = aie.buffer(tile_0_2, (tile_m_B, tile_n_B), T.i32())
        # out
        buffer_0_2_c = aie.buffer(tile_0_2, (tile_m_C, tile_n_C), T.i32())

        # input
        lock_0_1_read_in_a = aie.lock(tile_0_1, lock_id=0, init=1)
        lock_0_1_write_out_a = aie.lock(tile_0_1, lock_id=1, init=0)

        lock_0_2_read_in_a = aie.lock(tile_0_2, lock_id=0, init=1)
        lock_0_2_use_a = aie.lock(tile_0_2, lock_id=1, init=0)
        lock_0_2_read_in_b = aie.lock(tile_0_2, lock_id=2, init=1)
        lock_0_2_use_b = aie.lock(tile_0_2, lock_id=3, init=0)
        lock_0_2_use_c = aie.lock(tile_0_2, lock_id=4, init=1)
        lock_0_2_write_out_c = aie.lock(tile_0_2, lock_id=5, init=0)

        # input flow
        # a
        aie.flow(tile_0_0, DMA, 0, tile_0_1, DMA, 0)
        aie.flow(tile_0_1, DMA, 0, tile_0_2, DMA, 0)
        # b
        aie.flow(tile_0_0, DMA, 1, tile_0_1, DMA, 1)
        aie.flow(tile_0_1, DMA, 1, tile_0_2, DMA, 1)
        # output flow
        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        col = 0
        # in A
        channel_index = 0
        ddr_id = 0
        offsets = [
            0,
            # A tiles are "fat" so need to offset by rows (i.e. d1 dim)
            0 + d1_size_A * d1_stride_A,
        ]
        for i, bd_id in enumerate(range(2)):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_A * tile_n_A,
                    buffer_offset=offsets[i],
                    ddr_id=ddr_id,
                )
            )
            npu_insts.extend(aiex.npu.write32(MM2S, channel_index, col, bd_id))

        # in B
        channel_index = 1
        col = 0
        ddr_id = 1
        for bd_id in range(bd_id + 1, bd_id + 1 + 4, 2):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    buffer_offset=0,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
            )
            npu_insts.extend(aiex.npu.write32(MM2S, channel_index, col, bd_id))
            bd_id += 1
            # B tiles are "tall" so need to offset by cols (i.e. d0 dim)
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_B * tile_n_B,
                    buffer_offset=d0_size_B * d0_stride_B,
                    ddr_id=ddr_id,
                    d1_size=d1_size_B,
                    d1_stride=d1_stride_B,
                    d0_size=d0_size_B,
                    d0_stride=d0_stride_B,
                )
            )
            npu_insts.extend(aiex.npu.write32(MM2S, channel_index, col, bd_id))

        # out C
        channel_index = 0
        col = 0
        ddr_id = 2
        offsets = [
            0,
            0 + d0_size_C * d0_stride_C,
            d1_size_C * d1_stride_C,
            d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
        ]

        for i, bd_id in enumerate(range(bd_id + 1, bd_id + 1 + 4)):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=tile_m_C * tile_n_C,
                    buffer_offset=offsets[i],
                    ddr_id=ddr_id,
                    d1_size=d1_size_C,
                    d1_stride=d1_stride_C,
                    d0_size=d0_size_C,
                    d0_stride=d0_stride_C,
                )
            )
            npu_insts.extend(aiex.npu.write32(S2MM, channel_index, col, bd_id))
            npu_insts.extend(
                aiex.npu.sync(
                    channel=0,
                    column=0,
                    column_num=1,
                    direction=0,
                    row=0,
                    row_num=1,
                )
            )

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            # input flow
            buffer_0_1_a = aie.buffer(tile_0_1, (tile_m_A, tile_n_A), T.i32())
            buffer_0_1_b = aie.buffer(tile_0_1, (tile_m_B, tile_n_B), T.i32())
            # output flow
            buffer_0_1_c = aie.buffer(tile_0_1, (tile_m_C, tile_n_C), T.i32())

            @aie.dma(S2MM, channel_index=0)
            def dma1():
                aiex.process_bd(lock_0_1_read_in_a, buffer_0_1_a, lock_0_1_write_out_a)

            @aie.dma(MM2S, channel_index=0, num_blocks=2)
            def dma2():
                aiex.process_bd(
                    lock_0_1_write_out_a, buffer_0_1_a, lock_0_1_write_out_a
                )

            @aie.another_bd(dma2)
            def dma2point5():
                aiex.process_bd(lock_0_1_write_out_a, buffer_0_1_a, lock_0_1_read_in_a)

            aiex.forward_bd(tile_0_1, buffer_0_1_b, 1)
            aiex.forward_bd(tile_0_1, buffer_0_1_c, 2)

            aie.end()

        @aie.mem(tile_0_2)
        def mem_0_2():
            # input
            @aie.dma(S2MM, channel_index=0)
            def dma1():
                aiex.process_bd(lock_0_2_read_in_a, buffer_0_2_a, lock_0_2_use_a)

            @aie.dma(S2MM, channel_index=1)
            def dma2():
                aiex.process_bd(lock_0_2_read_in_b, buffer_0_2_b, lock_0_2_use_b)

            # output
            @aie.dma(MM2S, channel_index=0)
            def dma3():
                aiex.process_bd(lock_0_2_write_out_c, buffer_0_2_c, lock_0_2_use_c)

            aie.end()

        @aie.core(tile_0_2)
        def core():
            for _ in range_(0, tile_rows_C):
                for _ in range_(0, tile_cols_C):
                    with (
                        aiex.hold_lock(lock_0_2_use_a, lock_0_2_read_in_a),
                        aiex.hold_lock(lock_0_2_use_b, lock_0_2_read_in_b),
                        aiex.hold_lock(lock_0_2_use_c, lock_0_2_write_out_c),
                    ):
                        linalg_fill(arith.constant(0), outs=[buffer_0_2_c])
                        matmul_i32_i32_already_vectorized(
                            buffer_0_2_a, buffer_0_2_b, buffer_0_2_c
                        )

                    yield_([])
                yield_([])

    mod_aie.finish()
    mod_aievec = ExplicitlyManagedModule()
    matmul_i32_i32_already_vectorized.emit(force=True)
    mod_aievec = mod_aievec.finish()

    compile_with_vectorization(mod_aie, mod_aievec, workdir)
    xclbin_path = make_xclbin(mod_aie, workdir)
    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_npu_instructions(npu_insts)
        wrap_A, wrap_B, wrap_C = map(
            np.asarray, xclbin.mmap_buffers([(M, N), (M, N), (M, N)], np.int32)
        )

        A = np.random.randint(0, 10, (M, N), dtype=np.int32)
        B = np.random.randint(0, 10, (M, N), dtype=np.int32)
        C = np.zeros((M, N), dtype=np.int32)

        np.copyto(wrap_A, A, casting="no")
        np.copyto(wrap_B, B, casting="no")
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(A @ B, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(A @ B)
                print(wrap_C)
                assert False
