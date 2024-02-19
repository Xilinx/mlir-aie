# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: VITIS_DIR=$VITIS WORKDIR=$PWD XRT_DIR=%XRT_DIR %PYTHON %s

from __future__ import annotations

from operator import attrgetter
import sys

from aie.extras.context import ExplicitlyManagedModule
from aie.extras.dialects.ext import arith, func, linalg, scf, vector
from filelock import FileLock
import numpy as np

from aie.dialects import aie, aievec, aiex
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    FlowEndPoint,
    LockAction,
    WireBundle,
)
import aie.extras.types as T
from aie.util import tiling_calculator_n_tiles
from aie.xrt import XCLBin
from util import (
    compile_with_vectorization,
    compile_without_vectorization,
    construct_and_print_module,
    make_xclbin,
)

range_ = scf.range_
yield_ = scf.yield_

DMA = WireBundle.DMA
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

(
    _,
    _,
    (d1_size_A, d1_stride_A),
    (d0_size_A, d0_stride_A),
) = tiling_calculator_n_tiles(M, N, n_tile_rows=tile_rows_A, n_tile_cols=tile_cols_A)
(
    _,
    _,
    (d1_size_B, d1_stride_B),
    (d0_size_B, d0_stride_B),
) = tiling_calculator_n_tiles(M, N, n_tile_rows=tile_rows_B, n_tile_cols=tile_cols_B)
(
    _,
    _,
    (d1_size_C, d1_stride_C),
    (d0_size_C, d0_stride_C),
) = tiling_calculator_n_tiles(M, N, n_tile_rows=tile_rows_C, n_tile_cols=tile_cols_C)


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
    (_, _, (d1_size, d1_stride), (d0_size, d0_stride),) = tiling_calculator_n_tiles(
        M, N, n_tile_rows=n_tile_rows, n_tile_cols=n_tile_cols
    )

    ipu_insts = aiex.ipu.writebd_shimtile(
        bd_id=bd_id,
        ddr_id=ddr_id,
        column=column,
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


# CHECK-LABEL: tiled_nonsquare_tile_spatial
@construct_and_print_module
def tiled_nonsquare_tile_spatial(module):
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        # col a0 (top row of matrix products)
        tiles = np.empty((6, 5), dtype=object)
        for col in [0, 1]:
            for row in [0, 1, 2, 3]:
                tiles[col, row] = aie.tile(col, row)
        for col in [2, 3]:
            for row in [0, 1]:
                tiles[col, row] = aie.tile(col, row)

        # broadcast a0
        broadcast_a0_flow_ep = tiles[0, 0] >> tiles[0, 1]
        broadcast_a00_flow_ep, broadcast_a01_flow_ep = broadcast_a0_flow_ep >> (
            tiles[0, 2],
            tiles[0, 3],
        )
        # broadcast a1
        broadcast_a1_flow_ep = tiles[1, 0] >> tiles[1, 1]
        broadcast_a10_flow_ep, broadcast_a11_flow_ep = broadcast_a1_flow_ep >> (
            tiles[1, 2],
            tiles[1, 3],
        )

        # broadcast b0
        broadcast_b0_flow_ep = tiles[0, 0] >> tiles[0, 1]
        broadcast_b00_flow_ep, broadcast_b01_flow_ep = broadcast_b0_flow_ep >> (
            tiles[0, 2],
            tiles[1, 2],
        )
        # broadcast b1
        broadcast_b1_flow_ep = tiles[1, 0] >> tiles[1, 1]
        broadcast_b10_flow_ep, broadcast_b11_flow_ep = broadcast_b1_flow_ep >> (
            tiles[0, 3],
            tiles[1, 3],
        )

        # fmt: off
        column = 0
        # broadcast a0
        ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_A, tile_cols_A, 0, column, MM2S, broadcast_a0_flow_ep.flow.source_channel, 0, 0))
        # broadcast b0
        ipu_insts.extend(shim_tensor_slice(M, N, tile_rows_B, tile_cols_B, 0, column, MM2S, broadcast_b0_flow_ep.flow.source_channel, 1, 1))

        column = 1
        # broadcast a1
        ipu_insts.extend(
            shim_tensor_slice(M, N, tile_rows_A, tile_cols_A, d1_size_A * d1_stride_A, column, MM2S, broadcast_a1_flow_ep.flow.source_channel, 0, 0)
        )
        # broadcast b1
        ipu_insts.extend(
            shim_tensor_slice(M, N, tile_rows_B, tile_cols_B, d0_size_B * d0_stride_B, column, MM2S, broadcast_b1_flow_ep.flow.source_channel, 1, 1)
        )
        # fmt: on

        @aie.memtile_dma(tiles[0, 1])
        def memtile_dma_0_1():
            buffer_0_1_a0 = aie.buffer(
                T.memref(tile_m_A, tile_n_A, T.i32()), tiles[0, 1]
            )
            buffer_0_1_b0 = aie.buffer(
                T.memref(tile_m_B, tile_n_B, T.i32()), tiles[0, 1]
            )

            aiex.forward_bd(
                tiles[0, 1],
                buffer_0_1_a0,
                s2mm_channel_idx=broadcast_a0_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_a01_flow_ep
                mm2s_channel_idx=broadcast_a00_flow_ep.flow.source_channel,
            )
            aiex.forward_bd(
                tiles[0, 1],
                buffer_0_1_b0,
                s2mm_channel_idx=broadcast_b0_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_b01_flow_ep
                mm2s_channel_idx=broadcast_b00_flow_ep.flow.source_channel,
            )

            aie.end()

        @aie.memtile_dma(tiles[1, 1])
        def memtile_dma_1_1():
            buffer_1_1_a1 = aie.buffer(
                T.memref(tile_m_A, tile_n_A, T.i32()), tiles[1, 1]
            )
            buffer_1_1_b1 = aie.buffer(
                T.memref(tile_m_B, tile_n_B, T.i32()), tiles[1, 1]
            )

            aiex.forward_bd(
                tiles[1, 1],
                buffer_1_1_a1,
                s2mm_channel_idx=broadcast_a1_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_a11_flow_ep
                mm2s_channel_idx=broadcast_a10_flow_ep.flow.source_channel,
            )
            aiex.forward_bd(
                tiles[1, 1],
                buffer_1_1_b1,
                s2mm_channel_idx=broadcast_b1_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_b11_flow_ep
                mm2s_channel_idx=broadcast_b10_flow_ep.flow.source_channel,
            )

            aie.end()

        core_tiles = [
            [tiles[0, 2], tiles[0, 3]],
            [tiles[1, 2], tiles[1, 3]],
        ]
        # [a0*b0, a0*b1]
        result_c00_flow_ep, result_c01_flow_ep = tiles[2, 1] << (
            tiles[0, 2],
            tiles[0, 3],
        )
        # [a1*b0, a1*b1]
        result_c10_flow_ep, result_c11_flow_ep = tiles[3, 1] << (
            tiles[1, 2],
            tiles[1, 3],
        )
        # fmt: off
        flows = [
            [
                [f.flow for f in (broadcast_a00_flow_ep, broadcast_b00_flow_ep, result_c00_flow_ep)],
                [f.flow for f in (broadcast_a01_flow_ep, broadcast_b01_flow_ep, result_c01_flow_ep)]
            ],
            [
                [f.flow for f in (broadcast_a10_flow_ep, broadcast_b10_flow_ep, result_c10_flow_ep)],
                [f.flow for f in (broadcast_a11_flow_ep, broadcast_b11_flow_ep, result_c11_flow_ep)]
            ],
        ]
        # fmt: on

        for i in range(tile_rows_C):
            for j in range(tile_cols_C):
                tile = core_tiles[i][j]
                buffer_a = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile)
                buffer_b = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile)
                buffer_c = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile)
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
            [f.flow for f in tiles[2, 0] << (tiles[2, 1], tiles[2, 1])],
            [f.flow for f in tiles[3, 0] << (tiles[3, 1], tiles[3, 1])],
        ]
        # fmt: on

        for i, c in enumerate([2, 3]):

            @aie.memtile_dma(tiles[c, 1])
            def memtile_dma_c_1():
                buffer_c00 = aie.buffer(
                    T.memref(tile_m_C, tile_n_C, T.i32()),
                    tiles[c, 1],
                    sym_name=f"buffer_{c}_1_c_left",
                )
                buffer_c01 = aie.buffer(
                    T.memref(tile_m_C, tile_n_C, T.i32()),
                    tiles[c, 1],
                    sym_name=f"buffer_{c}_1_c_right",
                )

                aiex.forward_bd(
                    tiles[c, 1],
                    buffer_c00,
                    s2mm_channel_idx=flows[i][0][2].dest_channel,
                    mm2s_channel_idx=shim_flows[i][0].source_channel,
                )
                aiex.forward_bd(
                    tiles[c, 1],
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

    compile_without_vectorization(module)
    xclbin_path = make_xclbin(module)
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


# CHECK-LABEL: tiled_nonsquare_tile_spatial_vectorized
@construct_and_print_module
def tiled_nonsquare_tile_spatial_vectorized(module):
    FlowEndPoint._reset_used_channels()
    ipu_insts = aiex.ipu.get_prolog()

    mod_aie = ExplicitlyManagedModule()

    @aie.device(AIEDevice.ipu)
    def ipu():
        matmul_i32_i32_already_vectorized.emit(decl=True)

        # col a0 (top row of matrix products)
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)
        tile_0_3 = aie.tile(0, 3)

        # col a1 (bottom row of matrix products)
        tile_1_0 = aie.tile(1, 0)
        tile_1_1 = aie.tile(1, 1)
        tile_1_2 = aie.tile(1, 2)
        tile_1_3 = aie.tile(1, 3)

        # result
        tile_2_0 = aie.tile(2, 0)
        tile_2_1 = aie.tile(2, 1)
        tile_3_0 = aie.tile(3, 0)
        tile_3_1 = aie.tile(3, 1)

        # broadcast a0
        broadcast_a0_flow_ep = tile_0_0 >> tile_0_1
        broadcast_a00_flow_ep, broadcast_a01_flow_ep = broadcast_a0_flow_ep >> (
            tile_0_2,
            tile_0_3,
        )
        # broadcast a1
        broadcast_a1_flow_ep = tile_1_0 >> tile_1_1
        broadcast_a10_flow_ep, broadcast_a11_flow_ep = broadcast_a1_flow_ep >> (
            tile_1_2,
            tile_1_3,
        )

        # broadcast b0
        broadcast_b0_flow_ep = tile_0_0 >> tile_0_1
        broadcast_b00_flow_ep, broadcast_b01_flow_ep = broadcast_b0_flow_ep >> (
            tile_0_2,
            tile_1_2,
        )
        # broadcast b1
        broadcast_b1_flow_ep = tile_1_0 >> tile_1_1
        broadcast_b10_flow_ep, broadcast_b11_flow_ep = broadcast_b1_flow_ep >> (
            tile_0_3,
            tile_1_3,
        )

        # [a0*b0, a0*b1]
        result_c00_flow_ep, result_c01_flow_ep = tile_2_1 << (tile_0_2, tile_0_3)
        # [a1*b0, a1*b1]
        result_c10_flow_ep, result_c11_flow_ep = tile_3_1 << (tile_1_2, tile_1_3)

        column = 0
        # broadcast a0
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=0,
                ddr_id=0,
                column=column,
                buffer_length=tile_m_A * tile_n_A,
                buffer_offset=0,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                MM2S, broadcast_a0_flow_ep.flow.source_channel, column, bd_id=0
            )
        )
        # broadcast b0
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=1,
                ddr_id=1,
                column=column,
                buffer_length=tile_m_B * tile_n_B,
                buffer_offset=0,
                d1_size=d1_size_B,
                d1_stride=d1_stride_B,
                d0_size=d0_size_B,
                d0_stride=d0_stride_B,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                MM2S, broadcast_b0_flow_ep.flow.source_channel, column, bd_id=1
            )
        )

        column = 1
        # broadcast a1
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=0,
                ddr_id=0,
                column=column,
                buffer_length=tile_m_A * tile_n_A,
                buffer_offset=d1_size_A * d1_stride_A,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                MM2S, broadcast_a1_flow_ep.flow.source_channel, column, bd_id=0
            )
        )
        # broadcast b1
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=1,
                ddr_id=1,
                column=column,
                buffer_length=tile_m_B * tile_n_B,
                buffer_offset=d0_size_B * d0_stride_B,
                d1_size=d1_size_B,
                d1_stride=d1_stride_B,
                d0_size=d0_size_B,
                d0_stride=d0_stride_B,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                MM2S, broadcast_b1_flow_ep.flow.source_channel, column, bd_id=1
            )
        )

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            buffer_0_1_a0 = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile_0_1)
            buffer_0_1_b0 = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile_0_1)

            aiex.forward_bd(
                tile_0_1,
                buffer_0_1_a0,
                s2mm_channel_idx=broadcast_a0_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_a01_flow_ep
                mm2s_channel_idx=broadcast_a00_flow_ep.flow.source_channel,
            )
            aiex.forward_bd(
                tile_0_1,
                buffer_0_1_b0,
                s2mm_channel_idx=broadcast_b0_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_b01_flow_ep
                mm2s_channel_idx=broadcast_b00_flow_ep.flow.source_channel,
            )

            aie.end()

        @aie.memtile_dma(tile_1_1)
        def memtile_dma_1_1():
            buffer_1_1_a1 = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile_1_1)
            buffer_1_1_b1 = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile_1_1)

            aiex.forward_bd(
                tile_1_1,
                buffer_1_1_a1,
                s2mm_channel_idx=broadcast_a1_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_a11_flow_ep
                mm2s_channel_idx=broadcast_a10_flow_ep.flow.source_channel,
            )
            aiex.forward_bd(
                tile_1_1,
                buffer_1_1_b1,
                s2mm_channel_idx=broadcast_b1_flow_ep.flow.dest_channel,
                # also includes/handles broadcast_b11_flow_ep
                mm2s_channel_idx=broadcast_b10_flow_ep.flow.source_channel,
            )

            aie.end()

        tiles = [
            [tile_0_2, tile_0_3],
            [tile_1_2, tile_1_3],
        ]
        flow_eps = [
            [
                (broadcast_a00_flow_ep, broadcast_b00_flow_ep, result_c00_flow_ep),
                (broadcast_a01_flow_ep, broadcast_b01_flow_ep, result_c01_flow_ep),
            ],
            [
                (broadcast_a10_flow_ep, broadcast_b10_flow_ep, result_c10_flow_ep),
                (broadcast_a11_flow_ep, broadcast_b11_flow_ep, result_c11_flow_ep),
            ],
        ]

        for i in range(tile_rows_C):
            for j in range(tile_cols_C):
                tile = tiles[i][j]
                buffer_a = aie.buffer(T.memref(tile_m_A, tile_n_A, T.i32()), tile)
                buffer_b = aie.buffer(T.memref(tile_m_B, tile_n_B, T.i32()), tile)
                buffer_c = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile)
                lock_read_in_a = aie.lock(tile, init=1)
                lock_use_a = aie.lock(tile, init=0)
                lock_read_in_b = aie.lock(tile, init=1)
                lock_use_b = aie.lock(tile, init=0)
                lock_use_c = aie.lock(tile, init=1)
                lock_write_out_c = aie.lock(tile, init=0)

                @aie.mem(tile)
                def mem():
                    a_flow_ep, b_flow_ep, c_flow_ep = flow_eps[i][j]

                    @aie.dma(S2MM, a_flow_ep.flow.dest_channel)
                    def dma1():
                        aiex.process_bd(lock_read_in_a, buffer_a, lock_use_a)

                    @aie.dma(S2MM, b_flow_ep.flow.dest_channel)
                    def dma2():
                        aiex.process_bd(lock_read_in_b, buffer_b, lock_use_b)

                    @aie.dma(MM2S, c_flow_ep.flow.source_channel)
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
                        matmul_i32_i32_already_vectorized(buffer_a, buffer_b, buffer_c)

        # out C
        result_c00_shim_flow_ep, result_c01_shim_flow_ep = tile_2_0 << (
            tile_2_1,
            tile_2_1,
        )
        result_c10_shim_flow_ep, result_c11_shim_flow_ep = tile_3_0 << (
            tile_3_1,
            tile_3_1,
        )

        @aie.memtile_dma(tile_2_1)
        def memtile_dma_2_1():
            buffer_c00 = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_2_1)
            buffer_c01 = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_2_1)

            aiex.forward_bd(
                tile_2_1,
                buffer_c00,
                s2mm_channel_idx=result_c00_flow_ep.flow.dest_channel,
                mm2s_channel_idx=result_c00_shim_flow_ep.flow.source_channel,
            )
            aiex.forward_bd(
                tile_2_1,
                buffer_c01,
                s2mm_channel_idx=result_c01_flow_ep.flow.dest_channel,
                mm2s_channel_idx=result_c01_shim_flow_ep.flow.source_channel,
            )

            aie.end()

        @aie.memtile_dma(tile_3_1)
        def memtile_dma_3_1():
            buffer_c10 = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_3_1)
            buffer_c11 = aie.buffer(T.memref(tile_m_C, tile_n_C, T.i32()), tile_3_1)

            aiex.forward_bd(
                tile_3_1,
                buffer_c10,
                s2mm_channel_idx=result_c10_flow_ep.flow.dest_channel,
                mm2s_channel_idx=result_c10_shim_flow_ep.flow.source_channel,
            )
            aiex.forward_bd(
                tile_3_1,
                buffer_c11,
                s2mm_channel_idx=result_c11_flow_ep.flow.dest_channel,
                mm2s_channel_idx=result_c11_shim_flow_ep.flow.source_channel,
            )

            aie.end()

        offsets = [
            0,
            0 + d0_size_C * d0_stride_C,
            d1_size_C * d1_stride_C,
            d1_size_C * d1_stride_C + d0_size_C * d0_stride_C,
        ]

        column = 2
        # c00
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=0,
                ddr_id=2,
                column=column,
                buffer_length=tile_m_C * tile_n_C,
                buffer_offset=offsets[0],
                d1_size=d1_size_C,
                d1_stride=d1_stride_C,
                d0_size=d0_size_C,
                d0_stride=d0_stride_C,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                S2MM, result_c00_shim_flow_ep.flow.dest_channel, column, bd_id=0
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=result_c00_shim_flow_ep.flow.dest_channel, column=column
            )
        )
        # c01
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=1,
                ddr_id=2,
                column=column,
                buffer_length=tile_m_C * tile_n_C,
                buffer_offset=offsets[1],
                d1_size=d1_size_C,
                d1_stride=d1_stride_C,
                d0_size=d0_size_C,
                d0_stride=d0_stride_C,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                S2MM, result_c01_shim_flow_ep.flow.dest_channel, column, bd_id=1
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=result_c01_shim_flow_ep.flow.dest_channel, column=column
            )
        )

        column = 3
        # c10
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=0,
                ddr_id=2,
                column=column,
                buffer_length=tile_m_C * tile_n_C,
                buffer_offset=offsets[2],
                d1_size=d1_size_C,
                d1_stride=d1_stride_C,
                d0_size=d0_size_C,
                d0_stride=d0_stride_C,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                S2MM, result_c10_shim_flow_ep.flow.dest_channel, column, bd_id=0
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=result_c10_shim_flow_ep.flow.dest_channel, column=column
            )
        )
        # c01
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                bd_id=1,
                ddr_id=2,
                column=column,
                buffer_length=tile_m_C * tile_n_C,
                buffer_offset=offsets[3],
                d1_size=d1_size_C,
                d1_stride=d1_stride_C,
                d0_size=d0_size_C,
                d0_stride=d0_stride_C,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                S2MM, result_c11_shim_flow_ep.flow.dest_channel, column, bd_id=1
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=result_c11_shim_flow_ep.flow.dest_channel, column=column
            )
        )

    mod_aie.finish()
    mod_aievec = ExplicitlyManagedModule()
    matmul_i32_i32_already_vectorized.emit(force=True)
    mod_aievec = mod_aievec.finish()

    compile_with_vectorization(mod_aie, mod_aievec)
    xclbin_path = make_xclbin(mod_aie)
    with FileLock("/tmp/ipu.lock"):

        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
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
