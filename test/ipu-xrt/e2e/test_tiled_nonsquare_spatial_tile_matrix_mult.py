# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.


from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys

from aie.compiler.aiecc.main import emit_design_kernel_json
from aie.compiler.util import (
    compile_with_vectorization,
    compile_without_vectorization,
    make_xclbin,
)
from aie.dialects import aie, aievec, aiex
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)
from aie.dialects.aiex import TileArray
from aie.extras.context import ExplicitlyManagedModule

# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref, scf, vector

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
import aie.extras.types as T
from aie.ir import UnitAttr
from aie.util import tiling_calculator_n_tiles
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np
import pytest

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")


range_ = scf.range_
yield_ = scf.yield_

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def shim_tensor_slice(
    M,
    N,
    n_tile_rows,
    n_tile_cols,
    buffer_offset,
    column,
    channel_dir,
    channel_index,
    bd_id,
    ddr_id,
):
    (_, _, (d1_size, d1_stride), (d0_size, d0_stride)) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=n_tile_rows, n_tile_cols=n_tile_cols
    )

    ipu_insts = aiex.ipu.writebd_shimtile(
        column=column,
        bd_id=bd_id,
        ddr_id=ddr_id,
        buffer_length=(M // n_tile_rows) * (N // n_tile_cols),
        buffer_offset=buffer_offset,
        d1_size=d1_size,
        d1_stride=d1_stride,
        d0_size=d0_size,
        d0_stride=d0_stride,
    )
    ipu_insts.extend(
        aiex.ipu.shimtile_push_queue(channel_dir, channel_index, column, bd_id=bd_id)
    )
    return ipu_insts


def shim_bd(direction, channel, buffer_length, column=0, bd_id=0, ddr_id=0):
    ipu_insts = []
    ipu_insts.extend(
        aiex.ipu.writebd_shimtile(
            column=column, bd_id=bd_id, ddr_id=ddr_id, buffer_length=buffer_length
        )
    )
    ipu_insts.extend(
        aiex.ipu.shimtile_push_queue(direction, channel, column, bd_id=bd_id)
    )
    return ipu_insts


def test_tiled_nonsquare_tile_spatial_2x2(ctx: MLIRContext, workdir: Path):
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

    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        # col a0 (top row of matrix products)
        tiles = np.empty((5, 6), dtype=object)
        for col in [0, 1]:
            for row in [0, 1, 2, 3]:
                tiles[col, row] = aie.tile(col, row)
        for col in [2, 3]:
            for row in [0, 1]:
                tiles[col, row] = aie.tile(col, row)
        tiles = TileArray(df=tiles)

        # broadcast a0
        broadcast_a0_flow_ep = tiles[0, 0] >> tiles[0, 1]
        broadcast_a00_flow_ep, broadcast_a01_flow_ep = tiles[0, 1] >> tiles[0, [2, 3]]
        # broadcast a1
        broadcast_a1_flow_ep = tiles[1, 0] >> tiles[1, 1]
        broadcast_a10_flow_ep, broadcast_a11_flow_ep = tiles[1, 1] >> tiles[1, [2, 3]]

        # broadcast b0
        broadcast_b0_flow_ep = tiles[0, 0] >> tiles[0, 1]
        broadcast_b00_flow_ep, broadcast_b01_flow_ep = tiles[0, 1] >> tiles[[0, 1], 2]
        # broadcast b1
        broadcast_b1_flow_ep = tiles[1, 0] >> tiles[1, 1]
        broadcast_b10_flow_ep, broadcast_b11_flow_ep = tiles[1, 1] >> tiles[[0, 1], 3]

        # fmt: off
        column = 0
        # broadcast a0
        ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_A, tile_cols_A, 0, column, MM2S, broadcast_a0_flow_ep.source_channel, 0, 0))
        # broadcast b0
        ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_B, tile_cols_B, 0, column, MM2S, broadcast_b0_flow_ep.source_channel, 1, 1))

        column = 1
        # broadcast a1
        ipu_insts.extend(
            shim_tensor_slice(M, N, tile_rows_A, tile_cols_A, d1_size_A * d1_stride_A, column, MM2S, broadcast_a1_flow_ep.source_channel, 0, 0)
        )
        # broadcast b1
        ipu_insts.extend(
            shim_tensor_slice(M, N, tile_rows_B, tile_cols_B, d0_size_B * d0_stride_B, column, MM2S, broadcast_b1_flow_ep.source_channel, 1, 1)
        )
        # fmt: on

        @aie.memtile_dma(tiles.df[0, 1])
        def memtile_dma_0_1():
            buffer_0_1_a0 = aie.buffer(tiles.df[0, 1], (tile_m_A, tile_n_A), T.i32())
            buffer_0_1_b0 = aie.buffer(tiles.df[0, 1], (tile_m_B, tile_n_B), T.i32())

            aiex.forward_bd(
                tiles.df[0, 1],
                buffer_0_1_a0,
                s2mm_channel_idx=broadcast_a0_flow_ep.dest_channel,
                # also includes/handles broadcast_a01_flow_ep
                mm2s_channel_idx=broadcast_a00_flow_ep.source_channel,
            )
            aiex.forward_bd(
                tiles.df[0, 1],
                buffer_0_1_b0,
                s2mm_channel_idx=broadcast_b0_flow_ep.dest_channel,
                # also includes/handles broadcast_b01_flow_ep
                mm2s_channel_idx=broadcast_b00_flow_ep.source_channel,
            )

            aie.end()

        @aie.memtile_dma(tiles.df[1, 1])
        def memtile_dma_1_1():
            buffer_1_1_a1 = aie.buffer(tiles.df[1, 1], (tile_m_A, tile_n_A), T.i32())
            buffer_1_1_b1 = aie.buffer(tiles.df[1, 1], (tile_m_B, tile_n_B), T.i32())

            aiex.forward_bd(
                tiles.df[1, 1],
                buffer_1_1_a1,
                s2mm_channel_idx=broadcast_a1_flow_ep.dest_channel,
                # also includes/handles broadcast_a11_flow_ep
                mm2s_channel_idx=broadcast_a10_flow_ep.source_channel,
            )
            aiex.forward_bd(
                tiles.df[1, 1],
                buffer_1_1_b1,
                s2mm_channel_idx=broadcast_b1_flow_ep.dest_channel,
                # also includes/handles broadcast_b11_flow_ep
                mm2s_channel_idx=broadcast_b10_flow_ep.source_channel,
            )

            aie.end()

        # [a0*b0, a0*b1]
        result_c00_flow_ep, result_c01_flow_ep = tiles[2, 1] << tiles[0, [2, 3]]
        # [a1*b0, a1*b1]
        result_c10_flow_ep, result_c11_flow_ep = tiles[3, 1] << tiles[1, [2, 3]]
        # fmt: off
        flows = [
            [
                [broadcast_a00_flow_ep, broadcast_b00_flow_ep, result_c00_flow_ep],
                [broadcast_a01_flow_ep, broadcast_b01_flow_ep, result_c01_flow_ep]
            ],
            [
                [broadcast_a10_flow_ep, broadcast_b10_flow_ep, result_c10_flow_ep],
                [broadcast_a11_flow_ep, broadcast_b11_flow_ep, result_c11_flow_ep]
            ],
        ]
        # fmt: on

        core_tiles = tiles[[0, 1], [2, 3]].df
        for i in range(tile_rows_C):
            for j in range(tile_cols_C):
                tile = core_tiles[i][j]
                buffer_a = aie.buffer(tile, (tile_m_A, tile_n_A), T.i32())
                buffer_b = aie.buffer(tile, (tile_m_B, tile_n_B), T.i32())
                buffer_c = aie.buffer(tile, (tile_m_C, tile_n_C), T.i32())
                lock_read_in_a = aie.lock(tile, init=1)
                lock_use_a = aie.lock(tile, init=0)
                lock_read_in_b = aie.lock(tile, init=1)
                lock_use_b = aie.lock(tile, init=0)
                lock_use_c = aie.lock(tile, init=1)
                lock_write_out_c = aie.lock(tile, init=0)

                @aie.mem(tile)
                def mem():
                    a_flow, b_flow, c_flow = flows[i][j]

                    @aie.dma(S2MM, a_flow.dest_channel)
                    def dma1():
                        aiex.process_bd(lock_read_in_a, buffer_a, lock_use_a)

                    @aie.dma(S2MM, b_flow.dest_channel)
                    def dma2():
                        aiex.process_bd(lock_read_in_b, buffer_b, lock_use_b)

                    @aie.dma(MM2S, c_flow.source_channel)
                    def dma3():
                        aiex.process_bd(lock_write_out_c, buffer_c, lock_use_c)

                    aie.end()

                @aie.core(tile)
                def core():
                    with (
                        aiex.hold_lock(lock_use_a, lock_read_in_a),
                        aiex.hold_lock(lock_use_b, lock_read_in_b),
                        aiex.hold_lock(lock_use_c, lock_write_out_c),
                    ):
                        linalg.fill(0, buffer_c)
                        linalg.matmul(buffer_a, buffer_b, buffer_c)

        # fmt: off
        shim_flows = [
            [tiles[2, 0] << tiles[2, 1], tiles[2, 0] << tiles[2, 1]],
            [tiles[3, 0] << tiles[3, 1], tiles[3, 0] << tiles[3, 1]]
        ]
        # fmt: on

        for i, c in enumerate([2, 3]):

            @aie.memtile_dma(tiles[c, 1].df)
            def memtile_dma_c_1():
                buffer_c00 = aie.buffer(
                    tiles[c, 1].df,
                    (tile_m_C, tile_n_C),
                    T.i32(),
                    name=f"buffer_{c}_1_c_left",
                )
                buffer_c01 = aie.buffer(
                    tiles[c, 1].df,
                    (tile_m_C, tile_n_C),
                    T.i32(),
                    name=f"buffer_{c}_1_c_right",
                )

                aiex.forward_bd(
                    tiles[c, 1].df,
                    buffer_c00,
                    s2mm_channel_idx=flows[i][0][2].dest_channel,
                    mm2s_channel_idx=shim_flows[i][0].source_channel,
                )
                aiex.forward_bd(
                    tiles[c, 1].df,
                    buffer_c01,
                    s2mm_channel_idx=flows[i][1][2].dest_channel,
                    mm2s_channel_idx=shim_flows[i][1].source_channel,
                )

                aie.end()

        offsets = [
            0,
            0 + d0_size_C * d0_stride_C,
            d1_size_C * d1_stride_C,
            d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
        ]
        channels = [
            (2, shim_flows[0][0].dest_channel, 0),
            (2, shim_flows[0][1].dest_channel, 1),
            (3, shim_flows[1][0].dest_channel, 0),
            (3, shim_flows[1][1].dest_channel, 1),
        ]

        # fmt: off
        for i, (column, channel, bd_id) in enumerate(channels):
            ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_C, tile_cols_C, offsets[i], column, S2MM, channel, bd_id, 2))
            ipu_insts.extend(aiex.ipu.sync(channel=channel, column=column))
        # fmt: on

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
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


M = N = 32

tile_rows_A, tile_cols_A = 2, 1
tile_rows_B, tile_cols_B = 1, 2
tile_rows_C, tile_cols_C = 2, 2

tile_m_A, tile_n_A = M // tile_rows_A, N // tile_cols_A
tile_m_B, tile_n_B = M // tile_rows_B, N // tile_cols_B
tile_m_C, tile_n_C = M // tile_rows_C, N // tile_cols_C


@func.func(sym_visibility="private")
def matmul_i32_i32_already_vectorized(
    A: T.memref(tile_m_A, tile_n_A, T.i32()),
    B: T.memref(tile_m_B, tile_n_B, T.i32()),
    C: T.memref(tile_m_C, tile_n_C, T.i32()),
):
    vec16int32 = T.vector(16, T.i32())
    vec16int64 = T.vector(16, T.i64())

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
                shift_round_sat,
                C,
                [j, c0],
                in_bounds=[True],
            )
            yield_([])
        yield_([])


def test_tiled_nonsquare_tile_spatial_2x2_vectorized(ctx: MLIRContext, workdir: Path):
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

    ipu_insts = aiex.ipu.get_prolog()

    mod_aievec = ExplicitlyManagedModule()
    kernel = matmul_i32_i32_already_vectorized.emit(force=True)
    kernel.attributes["aie_kernel"] = UnitAttr.get()
    mod_aievec = mod_aievec.finish()

    mod_aie = ExplicitlyManagedModule()

    @aie.device(AIEDevice.ipu)
    def ipu():
        matmul_i32_i32_already_vectorized.emit(decl=True)
        # col a0 (top row of matrix products)
        tiles = np.empty((5, 6), dtype=object)
        for col in [0, 1]:
            for row in [0, 1, 2, 3]:
                tiles[col, row] = aie.tile(col, row)
        for col in [2, 3]:
            for row in [0, 1]:
                tiles[col, row] = aie.tile(col, row)
        tiles = TileArray(df=tiles)

        # broadcast a0
        broadcast_a0_flow_ep = tiles[0, 0] >> tiles[0, 1]
        broadcast_a00_flow_ep, broadcast_a01_flow_ep = tiles[0, 1] >> tiles[0, [2, 3]]
        # broadcast a1
        broadcast_a1_flow_ep = tiles[1, 0] >> tiles[1, 1]
        broadcast_a10_flow_ep, broadcast_a11_flow_ep = tiles[1, 1] >> tiles[1, [2, 3]]

        # broadcast b0
        broadcast_b0_flow_ep = tiles[0, 0] >> tiles[0, 1]
        broadcast_b00_flow_ep, broadcast_b01_flow_ep = tiles[0, 1] >> tiles[[0, 1], 2]
        # broadcast b1
        broadcast_b1_flow_ep = tiles[1, 0] >> tiles[1, 1]
        broadcast_b10_flow_ep, broadcast_b11_flow_ep = tiles[1, 1] >> tiles[[0, 1], 3]

        # fmt: off
        column = 0
        # broadcast a0
        ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_A, tile_cols_A, 0, column, MM2S, broadcast_a0_flow_ep.source_channel, 0, 0))
        # broadcast b0
        ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_B, tile_cols_B, 0, column, MM2S, broadcast_b0_flow_ep.source_channel, 1, 1))

        column = 1
        # broadcast a1
        ipu_insts.extend(
            shim_tensor_slice(M, N, tile_rows_A, tile_cols_A, d1_size_A * d1_stride_A, column, MM2S, broadcast_a1_flow_ep.source_channel, 0, 0)
        )
        # broadcast b1
        ipu_insts.extend(
            shim_tensor_slice(M, N, tile_rows_B, tile_cols_B, d0_size_B * d0_stride_B, column, MM2S, broadcast_b1_flow_ep.source_channel, 1, 1)
        )
        # fmt: on

        @aie.memtile_dma(tiles.df[0, 1])
        def memtile_dma_0_1():
            buffer_0_1_a0 = aie.buffer(tiles.df[0, 1], (tile_m_A, tile_n_A), T.i32())
            buffer_0_1_b0 = aie.buffer(tiles.df[0, 1], (tile_m_B, tile_n_B), T.i32())

            aiex.forward_bd(
                tiles.df[0, 1],
                buffer_0_1_a0,
                s2mm_channel_idx=broadcast_a0_flow_ep.dest_channel,
                # also includes/handles broadcast_a01_flow_ep
                mm2s_channel_idx=broadcast_a00_flow_ep.source_channel,
            )
            aiex.forward_bd(
                tiles.df[0, 1],
                buffer_0_1_b0,
                s2mm_channel_idx=broadcast_b0_flow_ep.dest_channel,
                # also includes/handles broadcast_b01_flow_ep
                mm2s_channel_idx=broadcast_b00_flow_ep.source_channel,
            )

            aie.end()

        @aie.memtile_dma(tiles.df[1, 1])
        def memtile_dma_1_1():
            buffer_1_1_a1 = aie.buffer(tiles.df[1, 1], (tile_m_A, tile_n_A), T.i32())
            buffer_1_1_b1 = aie.buffer(tiles.df[1, 1], (tile_m_B, tile_n_B), T.i32())

            aiex.forward_bd(
                tiles.df[1, 1],
                buffer_1_1_a1,
                s2mm_channel_idx=broadcast_a1_flow_ep.dest_channel,
                # also includes/handles broadcast_a11_flow_ep
                mm2s_channel_idx=broadcast_a10_flow_ep.source_channel,
            )
            aiex.forward_bd(
                tiles.df[1, 1],
                buffer_1_1_b1,
                s2mm_channel_idx=broadcast_b1_flow_ep.dest_channel,
                # also includes/handles broadcast_b11_flow_ep
                mm2s_channel_idx=broadcast_b10_flow_ep.source_channel,
            )

            aie.end()

        # [a0*b0, a0*b1]
        result_c00_flow_ep, result_c01_flow_ep = tiles[2, 1] << tiles[0, [2, 3]]
        # [a1*b0, a1*b1]
        result_c10_flow_ep, result_c11_flow_ep = tiles[3, 1] << tiles[1, [2, 3]]
        # fmt: off
        flows = [
            [
                [broadcast_a00_flow_ep, broadcast_b00_flow_ep, result_c00_flow_ep],
                [broadcast_a01_flow_ep, broadcast_b01_flow_ep, result_c01_flow_ep]
            ],
            [
                [broadcast_a10_flow_ep, broadcast_b10_flow_ep, result_c10_flow_ep],
                [broadcast_a11_flow_ep, broadcast_b11_flow_ep, result_c11_flow_ep]
            ],
        ]
        # fmt: on

        core_tiles = tiles[[0, 1], [2, 3]].df
        for i in range(tile_rows_C):
            for j in range(tile_cols_C):
                tile = core_tiles[i][j]
                buffer_a = aie.buffer(tile, (tile_m_A, tile_n_A), T.i32())
                buffer_b = aie.buffer(tile, (tile_m_B, tile_n_B), T.i32())
                buffer_c = aie.buffer(tile, (tile_m_C, tile_n_C), T.i32())
                lock_read_in_a = aie.lock(tile, init=1)
                lock_use_a = aie.lock(tile, init=0)
                lock_read_in_b = aie.lock(tile, init=1)
                lock_use_b = aie.lock(tile, init=0)
                lock_use_c = aie.lock(tile, init=1)
                lock_write_out_c = aie.lock(tile, init=0)

                @aie.mem(tile)
                def mem():
                    a_flow, b_flow, c_flow = flows[i][j]

                    @aie.dma(S2MM, a_flow.dest_channel)
                    def dma1():
                        aiex.process_bd(lock_read_in_a, buffer_a, lock_use_a)

                    @aie.dma(S2MM, b_flow.dest_channel)
                    def dma2():
                        aiex.process_bd(lock_read_in_b, buffer_b, lock_use_b)

                    @aie.dma(MM2S, c_flow.source_channel)
                    def dma3():
                        aiex.process_bd(lock_write_out_c, buffer_c, lock_use_c)

                    aie.end()

                @aie.core(tile, link_with=f"{kernel.sym_name.value}.o")
                def core():
                    with (
                        aiex.hold_lock(lock_use_a, lock_read_in_a),
                        aiex.hold_lock(lock_use_b, lock_read_in_b),
                        aiex.hold_lock(lock_use_c, lock_write_out_c),
                    ):
                        linalg.fill(0, buffer_c)
                        linalg.matmul(buffer_a, buffer_b, buffer_c)

        # fmt: off
        shim_flows = [
            [tiles[2, 0] << tiles[2, 1], tiles[2, 0] << tiles[2, 1]],
            [tiles[3, 0] << tiles[3, 1], tiles[3, 0] << tiles[3, 1]]
        ]
        # fmt: on

        for i, c in enumerate([2, 3]):

            @aie.memtile_dma(tiles[c, 1].df)
            def memtile_dma_c_1():
                buffer_c00 = aie.buffer(
                    tiles[c, 1].df,
                    (tile_m_C, tile_n_C),
                    T.i32(),
                    name=f"buffer_{c}_1_c_left",
                )
                buffer_c01 = aie.buffer(
                    tiles[c, 1].df,
                    (tile_m_C, tile_n_C),
                    T.i32(),
                    name=f"buffer_{c}_1_c_right",
                )

                aiex.forward_bd(
                    tiles[c, 1].df,
                    buffer_c00,
                    s2mm_channel_idx=flows[i][0][2].dest_channel,
                    mm2s_channel_idx=shim_flows[i][0].source_channel,
                )
                aiex.forward_bd(
                    tiles[c, 1].df,
                    buffer_c01,
                    s2mm_channel_idx=flows[i][1][2].dest_channel,
                    mm2s_channel_idx=shim_flows[i][1].source_channel,
                )

                aie.end()

        offsets = [
            0,
            0 + d0_size_C * d0_stride_C,
            d1_size_C * d1_stride_C,
            d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
        ]
        channels = [
            (2, shim_flows[0][0].dest_channel, 0),
            (2, shim_flows[0][1].dest_channel, 1),
            (3, shim_flows[1][0].dest_channel, 0),
            (3, shim_flows[1][1].dest_channel, 1),
        ]

        # fmt: off
        for i, (column, channel, bd_id) in enumerate(channels):
            ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_C, tile_cols_C, offsets[i], column, S2MM, channel, bd_id, 2))
            ipu_insts.extend(aiex.ipu.sync(channel=channel, column=column))
        # fmt: on

    mod_aie = mod_aie.finish()

    compile_with_vectorization(mod_aie, mod_aievec, workdir)

    xclbin_path = make_xclbin(mod_aie, workdir)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
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


def test_tiled_nonsquare_tile_spatial_4x4_weight_stationary_v1(
    ctx: MLIRContext, workdir: Path
):
    K = 32
    cols = [0, 1, 2, 3]
    rows = [0, 1, 2, 3, 4, 5]

    dest_channels = {}

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols, rows)
        for i, t in enumerate(tiles[:, 2:]):
            b = aie.buffer(
                t.tile,
                (K,),
                T.i32(),
                initial_value=np.full((K,), i, dtype=np.int32),
            )
            aie.lock(t.tile, init=0)

        for t in tiles[:, 5]:
            self_lock = t.locks()[0]
            self_buffer = t.buffers()[0]

            @aie.core(t.tile)
            def core():
                with aiex.hold_lock(self_lock, self_lock, acq_val=0):
                    linalg.add(self_buffer, self_buffer, self_buffer)

        for t in tiles[:, 2:5]:
            north_lock = t.neighbors().north.locks()[0]
            north_buffer = t.neighbors().north.buffers()[0]

            self_lock = t.locks()[0]
            self_buffer = t.buffers()[0]

            @aie.core(t.tile)
            def core():
                with (
                    aiex.hold_lock(self_lock, self_lock, acq_val=0),
                    aiex.hold_lock(north_lock, north_lock),
                ):
                    linalg.add(north_buffer, self_buffer, self_buffer)

        tiles[:, 2] >> tiles[:, 1]

        for t in tiles[:, 2]:
            self_lock = t.locks()[0]
            self_buffer = t.buffers()[0]
            flow = t.flows()[0]

            @aie.mem(t.tile)
            def mem():
                aiex.send_bd(flow.source_channel, self_lock, self_buffer, self_lock)
                aie.end()

        tiles[:, 1] >> tiles[:, 0]

        for t in tiles[:, 1]:
            to_shim, to_mem = t.flows()

            @aie.memtile_dma(t.tile)
            def memtile_dma():
                buffer = aie.buffer(t.tile, (K,), T.i32())

                aiex.forward_bd(
                    t.tile,
                    buffer,
                    s2mm_channel_idx=to_mem.dest_channel,
                    mm2s_channel_idx=to_shim.source_channel,
                )
                aie.end()

        host_buffer = aie.external_buffer((K,), T.i32(), name=False)
        for t in tiles[:, 0]:
            flow = t.flows()[0]

            shim_start_lock = aie.lock(t.tile, init=0, sym_name=False)

            @aie.shim_dma(t.tile)
            def shim_dma_c():
                aiex.receive_bd(
                    int(flow.dest_channel),
                    shim_start_lock,
                    host_buffer,
                    acq_action=Acquire,
                    rel_val=0,
                    repeat_count=4,
                )

                aie.end()

            dest_channels[int(t.tile.col)] = int(flow.dest_channel)

    compile_without_vectorization(ctx.module, workdir, enable_cores=False)
    buffer_args = [f"out_col_{c}" for c in cols]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)
    ipu_insts = aiex.ipu.get_prolog()

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * len(cols), np.int32)

        bd_id = 0
        for i, col in enumerate(cols):
            ipu_insts.extend(
                aiex.ipu._update_tensor_addr_shim_tile(
                    col, bd_id, xclbin._get_buffer_host_address(i)
                )
            )
            ipu_insts.extend(aiex.ipu.lock_release(col, lock_id=0, lock_val=1))
            for r in rows:
                if r in {0, 1}:
                    continue
                ipu_insts.extend(aiex.ipu.enable_cores(col, r))
        for col in cols:
            dest_channel = dest_channels[col]
            ipu_insts.extend(aiex.ipu.sync(column=col, channel=dest_channel))

        xclbin.load_ipu_instructions(ipu_insts)

        wraps = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for w in wraps:
                print(w)


def test_tiled_nonsquare_tile_spatial_4x4_weight_stationary_5_cols(
    ctx: MLIRContext, workdir: Path
):
    K = 32
    cols = [0, 1, 2, 3, 4]
    rows = [0, 1, 2, 3, 4, 5]

    dest_channels = {}

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols, rows)
        tiles[0, 1].tile.erase()

        for i, t in enumerate(tiles[:, 2:]):
            b = aie.buffer(
                t.tile,
                (K,),
                T.i32(),
                initial_value=np.full((K,), i, dtype=np.int32),
            )
            aie.lock(t.tile, init=0)

        for t in tiles[:, 5]:
            self_lock = t.locks()[0]
            self_buffer = t.buffers()[0]

            @aie.core(t.tile)
            def core():
                with aiex.hold_lock(self_lock, self_lock, acq_val=0):
                    linalg.add(self_buffer, self_buffer, self_buffer)

        for t in tiles[:, 2:5]:
            north_lock = t.neighbors().north.locks()[0]
            north_buffer = t.neighbors().north.buffers()[0]

            self_lock = t.locks()[0]
            self_buffer = t.buffers()[0]

            @aie.core(t.tile)
            def core():
                with (
                    aiex.hold_lock(self_lock, self_lock, acq_val=0),
                    aiex.hold_lock(north_lock, north_lock),
                ):
                    linalg.add(north_buffer, self_buffer, self_buffer)

        tiles[1:, 2] >> tiles[1:, 1]
        # tiles[0, 2] >> tiles[0, 0]

        for t in tiles[1:, 2]:
            self_lock = t.locks()[0]
            self_buffer = t.buffers()[0]
            flow = t.flows()[0]

            @aie.mem(t.tile)
            def mem():
                aiex.send_bd(flow.source_channel, self_lock, self_buffer, self_lock)
                aie.end()

        tiles[1:, 1] >> tiles[1:, 0]

        for t in tiles[1:, 1]:
            to_shim, to_mem = t.flows()

            @aie.memtile_dma(t.tile)
            def memtile_dma():
                buffer = aie.buffer(t.tile, (K,), T.i32())

                aiex.forward_bd(
                    t.tile,
                    buffer,
                    s2mm_channel_idx=to_mem.dest_channel,
                    mm2s_channel_idx=to_shim.source_channel,
                )
                aie.end()

        host_buffer = aie.external_buffer((K,), T.i32(), name=False)
        for t in tiles[1:, 0]:
            flow = t.flows()[0]

            shim_start_lock = aie.lock(t.tile, init=0, sym_name=False)

            @aie.shim_dma(t.tile)
            def shim_dma_c():
                aiex.receive_bd(
                    int(flow.dest_channel),
                    shim_start_lock,
                    host_buffer,
                    acq_action=Acquire,
                    rel_val=0,
                    repeat_count=4,
                )

                aie.end()

            dest_channels[int(t.tile.col)] = int(flow.dest_channel)

    compile_without_vectorization(
        ctx.module, workdir, enable_cores=False, partition_start_col=0
    )
    buffer_args = [f"out_col_{c}" for c in cols[1:]]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(
        ctx.module, workdir, kernel_json=kernel_json, start_columns=[0]
    )
    ipu_insts = aiex.ipu.get_prolog()

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * (len(cols) - 1), np.int32)

        bd_id = 0
        for i, col in enumerate(cols[1:]):
            ipu_insts.extend(
                aiex.ipu._update_tensor_addr_shim_tile(
                    col, bd_id, xclbin._get_buffer_host_address(i)
                )
            )
            ipu_insts.extend(aiex.ipu.lock_release(col, lock_id=0, lock_val=1))
            for r in rows:
                if r in {0, 1}:
                    continue
                ipu_insts.extend(aiex.ipu.enable_cores(col, r))
        for col in cols[1:]:
            dest_channel = dest_channels[col]
            ipu_insts.extend(aiex.ipu.sync(column=col, channel=dest_channel))

        xclbin.load_ipu_instructions(ipu_insts)

        wraps = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for w in wraps:
                print(w)


def test_tiled_nonsquare_tile_spatial_4x4_broadcast(ctx: MLIRContext, workdir: Path):
    K = 32
    cols = [0, 1, 2, 3]
    rows = [0, 1, 2, 3, 4, 5]

    shim_channels = {}
    iters = 20

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols, rows)
        aie.slsl("shim to mem a")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="a", dest_annot="a")
        aie.slsl("shim to mem b")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="b", dest_annot="b")

        # do this in a loop instead of broadcasting to preserve comments in the right places..
        for col in cols:
            # broadcast out to the row
            aie.slsl(f"col={col} send a to col {col}")
            tiles[col, 1].flow(tiles[col, 2:], source_annot="a", dest_annot="a")

        for col in cols:
            # broadcast out to the col
            aie.slsl(f"col={col} send b to row {col + 2}")
            tiles[col, 1].flow(tiles[:, col + 2], source_annot="b", dest_annot="b")

        for col in cols:
            # get result back
            aie.slsl(f"col={col} return c from col {col}")
            # skip col 4, 5 for now (not enough incoming on memtiles and
            # can't route directly to shim tiles - router flubs and connects south to south???)
            tiles[col, 1].rflow(tiles[col, 2::2], source_annot="c", dest_annot="c")
            # connect 5th row directly to shim because mem tile is out of connections
            # tiles[col, 0].rflow(tiles[col, 4], source_annot="c", dest_annot="c_1")

        aie.slsl("mem to shim")
        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")
        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")

        for t in tiles[cols, 0]:
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_1_fl, in_c_2_fl = t.flows(filter_dest=True, dest_annot="c")
            shim_channels[int(t.tile.col), 0] = int(out_a_fl.source_channel)
            shim_channels[int(t.tile.col), 1] = int(out_b_fl.source_channel)
            shim_channels[int(t.tile.col), 2] = int(in_c_1_fl.dest_channel)
            shim_channels[int(t.tile.col), 3] = int(in_c_2_fl.dest_channel)

        aie.slsl("configure mem tile dmas")
        for t in tiles[cols, 1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_1_fl, in_c_2_fl = t.flows(filter_dest=True, dest_annot="c")
            out_c_1_fl, out_c_2_fl = t.flows(filter_source=True, source_annot="c")

            @aie.memtile_dma(t.tile)
            def mem():
                a_buffer = aie.buffer(t.tile, (K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    a_buffer,
                    s2mm_channel_idx=in_a_fl.dest_channel,
                    mm2s_channel_idx=out_a_fl.source_channel,
                    repeat_count=iters - 1,
                )
                b_buffer = aie.buffer(t.tile, (K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    b_buffer,
                    s2mm_channel_idx=in_b_fl.dest_channel,
                    mm2s_channel_idx=out_b_fl.source_channel,
                    repeat_count=iters - 1,
                )
                c_1_buffer = aie.buffer(t.tile, (2 * K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    c_1_buffer,
                    s2mm_channel_idx=in_c_1_fl.dest_channel,
                    mm2s_channel_idx=out_c_1_fl.source_channel,
                    repeat_count=iters - 1,
                )
                c_2_buffer = aie.buffer(t.tile, (2 * K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    c_2_buffer,
                    s2mm_channel_idx=in_c_2_fl.dest_channel,
                    mm2s_channel_idx=out_c_2_fl.source_channel,
                    repeat_count=iters - 1,
                )

                aie.end()

        aie.slsl("configure core mem dmas")
        # reverse so that northern neighbors are done before
        for t in list(tiles[cols, 2:])[::-1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_c_fl = t.flows(filter_source=True, source_annot="c", single=True)

            in_a_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
            in_a_cons_lock = aie.lock(t.tile, init=0, sym_name=False)
            in_b_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
            in_b_cons_lock = aie.lock(t.tile, init=0, sym_name=False)
            if out_c_fl is not None:
                out_c_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
                out_c_cons_lock = aie.lock(t.tile, init=0, sym_name=False)

            a_buffer = t.buffer([(K,)], T.i32(), annot="a")
            b_buffer = t.buffer([(K,)], T.i32(), annot="b")
            # even row - recipient of odd row results
            if out_c_fl is not None:
                c_buffer = t.buffer([(2 * K,)], T.i32(), annot="c")
                north_c_buffer = t.neighbors().north.buffers(annot="c", single=True)
            else:
                c_buffer = aie.buffer(t.tile, (K,), T.i32(), annot="c")

            @aie.mem(t.tile)
            def mem():
                aiex.receive_bd(
                    int(in_a_fl.dest_channel),
                    in_a_prod_lock,
                    a_buffer,
                    in_a_cons_lock,
                    repeat_count=iters - 1,
                )
                aiex.receive_bd(
                    int(in_b_fl.dest_channel),
                    in_b_prod_lock,
                    b_buffer,
                    in_b_cons_lock,
                    repeat_count=iters - 1,
                )
                if out_c_fl is not None:
                    aiex.send_bd(
                        int(out_c_fl.source_channel),
                        out_c_cons_lock,
                        c_buffer,
                        out_c_prod_lock,
                        repeat_count=iters - 1,
                        len=2 * K,
                    )

                aie.end()

            @aie.core(t.tile)
            def core():
                y = memref.alloc(K, T.i32())
                if out_c_fl is not None:
                    c_own_slice = c_buffer[:K]
                    c_north_neigbhor_slice = c_buffer[K:]
                else:
                    c_own_slice = c_buffer
                for i in range_(iters):
                    linalg.fill(0, y)
                    with ExitStack() as stack:
                        stack.enter_context(
                            aiex.hold_lock(in_a_cons_lock, in_a_prod_lock)
                        )
                        stack.enter_context(
                            aiex.hold_lock(in_b_cons_lock, in_b_prod_lock)
                        )
                        if out_c_fl is not None:
                            stack.enter_context(
                                aiex.hold_lock(out_c_prod_lock, out_c_cons_lock)
                            )
                        linalg.add(a_buffer, b_buffer, c_own_slice)
                        linalg.add(y, c_own_slice, c_own_slice)
                        if out_c_fl is not None:
                            # TODO(max): right now this isn't synced at all but still works
                            linalg.copy(north_c_buffer, c_north_neigbhor_slice)
                    yield_()

    compile_without_vectorization(ctx.module, workdir)
    buffer_args = list(
        zip(
            [f"col_{c}_a" for c in cols],
            [f"col_{c}_b" for c in cols],
            [f"col_{c}_c_1" for c in cols],
            [f"col_{c}_c_2" for c in cols],
        )
    )
    buffer_args = [a for col in buffer_args for a in col]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    ipu_insts = aiex.ipu.get_prolog()
    bd_id_direction = {0: MM2S, 1: MM2S, 2: S2MM, 3: S2MM}

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        buffer_lengths = []
        for a in buffer_args:
            if "_c_" in a:
                buffer_lengths.append((2 * K,))
            else:
                buffer_lengths.append((K,))
        views = xclbin.mmap_buffers(buffer_lengths, np.int32)
        buffer_idx = -1
        for col in cols:
            for bd_id in [0, 1, 2, 3]:
                buffer_idx += 1
                if bd_id in {0, 1}:
                    writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                        column=col, bd_id=bd_id, buffer_length=K
                    )
                else:
                    writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                        column=col, bd_id=bd_id, buffer_length=2 * K
                    )
                ipu_insts.extend(
                    aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                        writebd_shimtile_insts,
                        tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=bd_id_direction[bd_id],
                        channel_index=shim_channels[col, bd_id],
                        column=col,
                        bd_id=bd_id,
                        repeats=iters - 1,
                    )
                )

        for col in cols:
            dest_channel = shim_channels[col, 2]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )
            dest_channel = shim_channels[col, 3]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)

        As = {c: np.random.randint(0, 10, (K,), dtype=np.int32) for c in cols}
        Bs = {c: np.random.randint(0, 10, (K,), dtype=np.int32) for c in cols}
        wraps = list(map(np.asarray, views))
        for col in cols:
            C1 = np.zeros((2 * K,), dtype=np.int32)
            C2 = np.zeros((2 * K,), dtype=np.int32)

            np.copyto(wraps[4 * col + 0], As[col], casting="no")
            np.copyto(wraps[4 * col + 1], Bs[col], casting="no")
            np.copyto(wraps[4 * col + 2], C1, casting="no")
            np.copyto(wraps[4 * col + 3], C2, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        wrap_r = []
        for col in cols:
            wrap_r.append(
                np.concatenate([wraps[4 * col + 2], wraps[4 * col + 3]]).reshape(4, 32)
            )
        wrap_r = np.concatenate(wrap_r).reshape(4, 4, 32)
        r = np.empty((len(As), len(Bs), 32), dtype=np.int32)
        for ci in cols:
            for cj in cols:
                r[ci, cj][:] = As[ci] + Bs[cj]

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            assert np.array_equal(wrap_r, r)


def test_tiled_nonsquare_tile_spatial_4x4_broadcast_use_all_6_memtile_dmas(
    ctx: MLIRContext, workdir: Path
):
    K = 32
    cols = [0, 1, 2, 3]
    rows = [0, 1, 2, 3, 4, 5]

    shim_channels = {}
    iters = 20
    BYTES_PER_INT32 = 4

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols, rows)
        aie.slsl("shim to mem a")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="a", dest_annot="a")
        aie.slsl("shim to mem b")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="b", dest_annot="b")

        # broadcast out to the row
        aie.slsl(f"send a to cols")
        tiles[:, 1].flow(tiles[:, 2:], source_annot="a", dest_annot="a")

        for col in cols:
            # broadcast out to the col
            aie.slsl(f"col={col} send b to row {col + 2}")
            tiles[col, 1].flow(tiles[:, col + 2], source_annot="b", dest_annot="b")

        for col in cols:
            # get result back
            aie.slsl(f"return c from col {col}")
            tiles[col, 1].rflow(tiles[col, 2:], source_annot="c", dest_annot="c")

        aie.slsl("mem to shim")
        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")
        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")

        for t in tiles[cols, 0]:
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_1_fl, in_c_2_fl = t.flows(filter_dest=True, dest_annot="c")
            shim_channels[int(t.tile.col), 0] = int(out_a_fl.source_channel)
            shim_channels[int(t.tile.col), 1] = int(out_b_fl.source_channel)
            shim_channels[int(t.tile.col), 2] = int(in_c_1_fl.dest_channel)
            shim_channels[int(t.tile.col), 3] = int(in_c_2_fl.dest_channel)

        aie.slsl("configure mem tile dmas")
        for t in tiles[cols, 1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_flows = t.flows(filter_dest=True, dest_annot="c")
            out_c_flows = t.flows(filter_source=True, source_annot="c")

            @aie.memtile_dma(t.tile)
            def mem():
                a_buffer = aie.buffer(
                    t.tile, (K,), T.i32(), name=f"memtile_{int(t.tile.col)}_a_buffer"
                )
                aiex.forward_bd(
                    t.tile,
                    a_buffer,
                    s2mm_channel_idx=in_a_fl.dest_channel,
                    mm2s_channel_idx=out_a_fl.source_channel,
                    repeat_count=iters - 1,
                )
                b_buffer = aie.buffer(
                    t.tile, (K,), T.i32(), name=f"memtile_{int(t.tile.col)}_b_buffer"
                )
                aiex.forward_bd(
                    t.tile,
                    b_buffer,
                    s2mm_channel_idx=in_b_fl.dest_channel,
                    mm2s_channel_idx=out_b_fl.source_channel,
                    repeat_count=iters - 1,
                )

                # split c_1, c_2 buffers amongst 4 input channels (1 for each row)
                c_1_buffer = aie.buffer(
                    t.tile,
                    (2 * K,),
                    T.i32(),
                    name=f"memtile_{int(t.tile.col)}_c_1_buffer",
                )
                c_2_buffer = aie.buffer(
                    t.tile,
                    (2 * K,),
                    T.i32(),
                    name=f"memtile_{int(t.tile.col)}_c_2_buffer",
                )
                in_c_read_in_locks = [
                    aie.lock(t.tile, init=1, sym_name=f"in_c_{i}_read_in_lock")
                    for i in range(len(in_c_flows))
                ]
                out_c_write_out_locks = [
                    aie.lock(t.tile, init=0, sym_name=f"out_c_{i}_write_out_lock")
                    for i in range(len(in_c_flows))
                ]

                for in_flow, read_in_lock, write_out_lock, buffer, offset in zip(
                    in_c_flows,
                    in_c_read_in_locks,
                    out_c_write_out_locks,
                    [c_1_buffer, c_1_buffer, c_2_buffer, c_2_buffer],
                    [0, K, 0, K],
                ):
                    aiex.receive_bd(
                        in_flow.dest_channel,
                        read_in_lock,
                        buffer,
                        write_out_lock,
                        len=K,
                        offset=offset * BYTES_PER_INT32,
                        repeat_count=iters - 1,
                    )
                for (
                    out_flow,
                    (left_half_write_out_lock, right_half_write_out_lock),
                    (left_half_read_in_lock, right_half_read_in_lock),
                    buffer,
                ) in zip(
                    out_c_flows,
                    [out_c_write_out_locks[:2], out_c_write_out_locks[2:]],
                    [in_c_read_in_locks[:2], in_c_read_in_locks[2:]],
                    [c_1_buffer, c_2_buffer],
                ):
                    dma = aiex.send_bd(
                        out_flow.source_channel,
                        left_half_write_out_lock,
                        buffer,
                        left_half_read_in_lock,
                        offset=0,
                        len=K,
                        repeat_count=iters - 1,
                        num_blocks=2,
                    )

                    @aie.another_bd(dma)
                    def second_half():
                        aiex.process_bd(
                            right_half_write_out_lock,
                            buffer,
                            right_half_read_in_lock,
                            offset=K * BYTES_PER_INT32,
                            len=K,
                        )

                aie.end()

        aie.slsl("configure core mem dmas")
        # reverse so that northern neighbors are done before
        for t in list(tiles[cols, 2:])[::-1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_c_fl = t.flows(filter_source=True, source_annot="c", single=True)

            in_a_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
            in_a_cons_lock = aie.lock(t.tile, init=0, sym_name=False)
            in_b_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
            in_b_cons_lock = aie.lock(t.tile, init=0, sym_name=False)
            out_c_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
            out_c_cons_lock = aie.lock(t.tile, init=0, sym_name=False)

            a_buffer = t.buffer([(K,)], T.i32(), annot="a")
            b_buffer = t.buffer([(K,)], T.i32(), annot="b")
            c_buffer = aie.buffer(t.tile, (K,), T.i32(), annot="c")

            @aie.mem(t.tile)
            def mem():
                aiex.receive_bd(
                    int(in_a_fl.dest_channel),
                    in_a_prod_lock,
                    a_buffer,
                    in_a_cons_lock,
                    repeat_count=iters - 1,
                )
                aiex.receive_bd(
                    int(in_b_fl.dest_channel),
                    in_b_prod_lock,
                    b_buffer,
                    in_b_cons_lock,
                    repeat_count=iters - 1,
                )
                aiex.send_bd(
                    int(out_c_fl.source_channel),
                    out_c_cons_lock,
                    c_buffer,
                    out_c_prod_lock,
                    repeat_count=iters - 1,
                )

                aie.end()

            @aie.core(t.tile)
            def core():
                y = memref.alloc(K, T.i32())
                linalg.fill(0, y)
                for i in range_(iters):
                    with (
                        aiex.hold_lock(in_a_cons_lock, in_a_prod_lock),
                        aiex.hold_lock(in_b_cons_lock, in_b_prod_lock),
                        aiex.hold_lock(out_c_prod_lock, out_c_cons_lock),
                    ):
                        linalg.add(a_buffer, b_buffer, c_buffer)
                    yield_()

    compile_without_vectorization(ctx.module, workdir)
    buffer_args = list(
        zip(
            [f"col_{c}_a" for c in cols],
            [f"col_{c}_b" for c in cols],
            [f"col_{c}_c_1" for c in cols],
            [f"col_{c}_c_2" for c in cols],
        )
    )
    buffer_args = [a for col in buffer_args for a in col]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    ipu_insts = aiex.ipu.get_prolog()
    bd_id_direction = {0: MM2S, 1: MM2S, 2: S2MM, 3: S2MM}

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        buffer_lengths = []
        for a in buffer_args:
            if "_c_" in a:
                buffer_lengths.append((2 * K,))
            else:
                buffer_lengths.append((K,))
        views = xclbin.mmap_buffers(buffer_lengths, np.int32)
        buffer_idx = -1
        for col in cols:
            for bd_id in [0, 1, 2, 3]:
                buffer_idx += 1
                if bd_id in {0, 1}:
                    writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                        column=col, bd_id=bd_id, buffer_length=K
                    )
                else:
                    writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                        column=col, bd_id=bd_id, buffer_length=2 * K
                    )
                ipu_insts.extend(
                    aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                        writebd_shimtile_insts,
                        tensor_addr=xclbin._get_buffer_host_address(buffer_idx),
                    )
                )
                ipu_insts.extend(
                    aiex.ipu.shimtile_push_queue(
                        channel_dir=bd_id_direction[bd_id],
                        channel_index=shim_channels[col, bd_id],
                        column=col,
                        bd_id=bd_id,
                        repeats=iters - 1,
                    )
                )

        for col in cols:
            dest_channel = shim_channels[col, 2]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )
            dest_channel = shim_channels[col, 3]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)

        As = {c: np.random.randint(0, 10, (K,), dtype=np.int32) for c in cols}
        Bs = {c: np.random.randint(0, 10, (K,), dtype=np.int32) for c in cols}
        wraps = list(map(np.asarray, views))
        for col in cols:
            C1 = np.zeros((2 * K,), dtype=np.int32)
            C2 = np.zeros((2 * K,), dtype=np.int32)

            np.copyto(wraps[4 * col + 0], As[col], casting="no")
            np.copyto(wraps[4 * col + 1], Bs[col], casting="no")
            np.copyto(wraps[4 * col + 2], C1, casting="no")
            np.copyto(wraps[4 * col + 3], C2, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        wrap_r = []
        for col in cols:
            wrap_r.append(
                np.concatenate([wraps[4 * col + 2], wraps[4 * col + 3]]).reshape(4, 32)
            )
        wrap_r = np.concatenate(wrap_r).reshape(4, 4, 32)
        r = np.empty((len(As), len(Bs), 32), dtype=np.int32)
        for ci in cols:
            for cj in cols:
                r[ci, cj][:] = As[ci] + Bs[cj]

        if not np.array_equal(wrap_r, r):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print("correct")
                print(r)
                print("output")
                print(wrap_r)
                assert False


def test_tiled_nonsquare_tile_spatial_4x4_broadcast_with_shim_dma_broken(
    ctx: MLIRContext, workdir: Path
):
    K = 32
    cols = [0, 1, 2, 3]
    rows = [0, 1, 2, 3, 4, 5]

    shim_channels = {}
    iters = 20

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols, rows)
        aie.slsl("shim to mem a")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="a", dest_annot="a")
        aie.slsl("shim to mem b")
        tiles[cols, 0].flow(tiles[cols, 1], source_annot="b", dest_annot="b")

        # do this in a loop instead of broadcasting to preserve comments in the right places..
        for col in cols:
            # broadcast out to the row
            aie.slsl(f"col={col} send a to col {col}")
            tiles[col, 1].flow(tiles[col, 2:], source_annot="a", dest_annot="a")

        for col in cols:
            # broadcast out to the col
            aie.slsl(f"col={col} send b to row {col + 2}")
            tiles[col, 1].flow(tiles[:, col + 2], source_annot="b", dest_annot="b")

        for col in cols:
            # get result back
            aie.slsl(f"col={col} return c from col {col}")
            # skip col 4, 5 for now (not enough incoming on memtiles and
            # can't route directly to shim tiles - router flubs and connects south to south???)
            tiles[col, 1].rflow(tiles[col, 2:4], source_annot="c", dest_annot="c")
            # connect 5th row directly to shim because mem tile is out of connections
            # tiles[col, 0].rflow(tiles[col, 4], source_annot="c", dest_annot="c_1")

        aie.slsl("mem to shim")
        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")
        tiles[cols, 1].flow(tiles[cols, 0], source_annot="c", dest_annot="c")

        for t in tiles[cols, 0]:
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_1_fl, in_c_2_fl = t.flows(filter_dest=True, dest_annot="c")

            shim_channels[int(t.tile.col), 0] = int(out_a_fl.source_channel)
            shim_channels[int(t.tile.col), 1] = int(out_b_fl.source_channel)
            shim_channels[int(t.tile.col), 2] = int(in_c_1_fl.dest_channel)
            shim_channels[int(t.tile.col), 3] = int(in_c_2_fl.dest_channel)

            host_buffer = aie.external_buffer((K,), T.i32(), name=False)

            for ch, dir in [
                (out_a_fl.source_channel, S2MM),
                (out_b_fl.source_channel, S2MM),
                (in_c_1_fl.dest_channel, MM2S),
                (in_c_2_fl.dest_channel, MM2S),
            ]:

                @aie.shim_dma(t.tile)
                def shim_dma_c_0():
                    shim_start_lock = aie.lock(t.tile, init=0, sym_name=False)

                    @aie.dma(dir, ch, loop=False, repeat_count=iters - 1)
                    def dma():
                        aie.use_lock(shim_start_lock, Acquire)
                        aie.dma_bd(host_buffer)
                        aie.use_lock(shim_start_lock, Release, value=0)

                    aie.end()

        aie.slsl("configure mem tile dmas")
        for t in tiles[cols, 1]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            out_a_fl = t.flows(filter_source=True, source_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_b_fl = t.flows(filter_source=True, source_annot="b", single=True)
            in_c_1_fl, in_c_2_fl = t.flows(filter_dest=True, dest_annot="c")
            out_c_1_fl, out_c_2_fl = t.flows(filter_source=True, source_annot="c")

            @aie.memtile_dma(t.tile)
            def mem():
                a_buffer = aie.buffer(t.tile, (K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    a_buffer,
                    s2mm_channel_idx=in_a_fl.dest_channel,
                    mm2s_channel_idx=out_a_fl.source_channel,
                    repeat_count=iters - 1,
                )
                b_buffer = aie.buffer(t.tile, (K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    b_buffer,
                    s2mm_channel_idx=in_b_fl.dest_channel,
                    mm2s_channel_idx=out_b_fl.source_channel,
                    repeat_count=iters - 1,
                )
                c_1_buffer = aie.buffer(t.tile, (K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    c_1_buffer,
                    s2mm_channel_idx=in_c_1_fl.dest_channel,
                    mm2s_channel_idx=out_c_1_fl.source_channel,
                    repeat_count=iters - 1,
                )
                c_2_buffer = aie.buffer(t.tile, (K,), T.i32())
                aiex.forward_bd(
                    t.tile,
                    c_2_buffer,
                    s2mm_channel_idx=in_c_2_fl.dest_channel,
                    mm2s_channel_idx=out_c_2_fl.source_channel,
                    repeat_count=iters - 1,
                )

                aie.end()

        aie.slsl("configure core mem dmas")
        for t in tiles[cols, 2:]:
            in_a_fl = t.flows(filter_dest=True, dest_annot="a", single=True)
            in_b_fl = t.flows(filter_dest=True, dest_annot="b", single=True)
            out_c_fl = t.flows(filter_source=True, source_annot="c", single=True)

            in_a_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
            in_a_cons_lock = aie.lock(t.tile, init=0, sym_name=False)
            in_b_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
            in_b_cons_lock = aie.lock(t.tile, init=0, sym_name=False)
            if out_c_fl is not None:
                out_c_prod_lock = aie.lock(t.tile, init=1, sym_name=False)
                out_c_cons_lock = aie.lock(t.tile, init=0, sym_name=False)

            a_buffer = aie.buffer(t.tile, (K,), T.i32())
            b_buffer = aie.buffer(t.tile, (K,), T.i32())
            c_buffer = aie.buffer(t.tile, (K,), T.i32())

            @aie.mem(t.tile)
            def mem():
                aiex.receive_bd(
                    int(in_a_fl.dest_channel),
                    in_a_prod_lock,
                    a_buffer,
                    in_a_cons_lock,
                    repeat_count=iters - 1,
                )
                aiex.receive_bd(
                    int(in_b_fl.dest_channel),
                    in_b_prod_lock,
                    b_buffer,
                    in_b_cons_lock,
                    repeat_count=iters - 1,
                )
                if out_c_fl is not None:
                    aiex.send_bd(
                        int(out_c_fl.source_channel),
                        out_c_cons_lock,
                        c_buffer,
                        out_c_prod_lock,
                        repeat_count=iters - 1,
                    )

                aie.end()

            @aie.core(t.tile)
            def core():
                y = memref.alloc(K, T.i32())
                for i in range_(iters):
                    linalg.fill(i, y)
                    with ExitStack() as stack:
                        stack.enter_context(
                            aiex.hold_lock(in_a_cons_lock, in_a_prod_lock)
                        )
                        stack.enter_context(
                            aiex.hold_lock(in_b_cons_lock, in_b_prod_lock)
                        )
                        if out_c_fl is not None:
                            stack.enter_context(
                                aiex.hold_lock(out_c_prod_lock, out_c_cons_lock)
                            )
                        linalg.add(a_buffer, b_buffer, c_buffer)
                        linalg.add(y, c_buffer, c_buffer)
                    yield_()

    enable_cores = False
    compile_without_vectorization(ctx.module, workdir, enable_cores=enable_cores)
    buffer_args = list(
        zip(
            [f"col_{c}_a" for c in cols],
            [f"col_{c}_b" for c in cols],
            [f"col_{c}_c_1" for c in cols],
            [f"col_{c}_c_2" for c in cols],
        )
    )
    buffer_args = [a for col in buffer_args for a in col]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    ipu_insts = aiex.ipu.get_prolog()

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * len(buffer_args), np.int32)
        buffer_idx = -1
        for col in cols:
            for bd_id in [0, 1, 2, 3]:
                buffer_idx += 1
                ipu_insts.extend(
                    aiex.ipu._update_tensor_addr_shim_tile(
                        col, bd_id, xclbin._get_buffer_host_address(buffer_idx)
                    )
                )

        # TODO(max): need to figureo out a way to disable the channel
        # not lock
        # runtime_lib\xaiengine\aie-rt\driver\src\global\xaiemlgbl_reginit.c:410
        for col in cols:
            for lock_id in range(4):
                ipu_insts.extend(
                    aiex.ipu.lock_release(col, lock_id=lock_id, lock_val=1)
                )

        for col in cols:
            for row in rows[2:]:
                ipu_insts.extend(aiex.ipu.enable_cores(col, row))

        for col in cols:
            dest_channel = shim_channels[col, 2]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )
            dest_channel = shim_channels[col, 3]
            ipu_insts.extend(
                aiex.ipu.sync(column=col, channel=dest_channel, direction=S2MM)
            )

        xclbin.load_ipu_instructions(ipu_insts)

        wraps = list(map(np.asarray, views))
        for col in cols:
            A = np.random.randint(0, 10, (K,), dtype=np.int32)
            B = np.random.randint(0, 10, (K,), dtype=np.int32)
            C1 = np.zeros((K,), dtype=np.int32)
            C2 = np.zeros((K,), dtype=np.int32)

            np.copyto(wraps[4 * col + 0], A, casting="no")
            np.copyto(wraps[4 * col + 1], B, casting="no")
            np.copyto(wraps[4 * col + 2], C1, casting="no")
            np.copyto(wraps[4 * col + 3], C2, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for i, w in enumerate(wraps):
                print(buffer_args[i], w)
