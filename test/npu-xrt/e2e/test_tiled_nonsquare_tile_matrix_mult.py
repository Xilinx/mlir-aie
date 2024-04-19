# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.


from pathlib import Path
import sys

from aie.compiler.util import (
    compile_without_vectorization,
    make_xclbin,
)
from aie.dialects import aie, aiex
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)
from aie.dialects.scf import for_ as range_, yield_
from aie.extras.dialects.ext import linalg

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
import aie.extras.types as T
from aie.util import tiling_calculator_n_tiles
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np
import pytest

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def test_tiled_nonsquare_tile_matrix_mult(ctx: MLIRContext, workdir: Path):
    M = N = 32

    tile_rows_A, tile_cols_A = 2, 1
    tile_rows_B, tile_cols_B = 1, 2
    tile_rows_C, tile_cols_C = 2, 2

    tile_m_A, tile_n_A = M // tile_rows_A, N // tile_cols_A
    tile_m_B, tile_n_B = M // tile_rows_B, N // tile_cols_B
    tile_m_C, tile_n_C = M // tile_rows_C, N // tile_cols_C

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

    @aie.device(AIEDevice.npu)
    def npu():
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
                    channel=0, column=0, column_num=1, direction=0, row=0, row_num=1
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

                    linalg.fill(0, buffer_0_2_c)
                    linalg.matmul(buffer_0_2_a, buffer_0_2_b, buffer_0_2_c)

                    aie.use_lock(lock_0_2_read_in_a, Release)
                    aie.use_lock(lock_0_2_read_in_b, Release)
                    aie.use_lock(lock_0_2_write_out_c, Release)
                    yield_([])
                yield_([])

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
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


def test_tiled_nonsquare_tile_matrix_mult_sugar(ctx: MLIRContext, workdir: Path):
    M = N = 32

    tile_rows_A, tile_cols_A = 2, 1
    tile_rows_B, tile_cols_B = 1, 2
    tile_rows_C, tile_cols_C = 2, 2

    tile_m_A, tile_n_A = M // tile_rows_A, N // tile_cols_A
    tile_m_B, tile_n_B = M // tile_rows_B, N // tile_cols_B
    tile_m_C, tile_n_C = M // tile_rows_C, N // tile_cols_C

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

    @aie.device(AIEDevice.npu)
    def npu():
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
                    channel=0, column=0, column_num=1, direction=0, row=0, row_num=1
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
                        linalg.fill(0, buffer_0_2_c)
                        linalg.matmul(buffer_0_2_a, buffer_0_2_b, buffer_0_2_c)
                    yield_([])
                yield_([])

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
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
