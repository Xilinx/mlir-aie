# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.


from __future__ import annotations

from pathlib import Path
import sys

# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref, scf, vector

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
from filelock import FileLock
import numpy as np
import pytest

from aie.compiler.aiecc.main import emit_design_kernel_json
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
    bd_dim_layout,
)
from aie.dialects.aiex import TileArray
import aie.extras.types as T
from aie.xrt import XCLBin

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


def test_double_pump_single_shim_buffer(ctx: MLIRContext, workdir: Path):
    K = 32
    player_a_channel = 0
    player_b_channel = 1
    iters = 10
    col = 0
    compute_tile_row = 2

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_2 = aie.tile(col, compute_tile_row)
        tile_0_0 = aie.tile(col, 0)
        output_buffer_a = aie.buffer(
            tile_0_2, (K,), T.i32(), initial_value=np.ones((K,), dtype=np.int32) * 3
        )
        channel_a_prod_lock = aie.lock(tile_0_2, init=1, sym_name=False)
        channel_a_cons_lock = aie.lock(tile_0_2, init=0, sym_name=False)

        channel_b_prod_lock = aie.lock(tile_0_2, init=1, sym_name=False)
        channel_b_cons_lock = aie.lock(tile_0_2, init=0, sym_name=False)
        output_buffer_b = aie.buffer(
            tile_0_2, (K,), T.i32(), initial_value=np.ones((K,), dtype=np.int32) * 4
        )

        aie.flow(tile_0_2, DMA, player_a_channel, tile_0_0, DMA, player_a_channel)
        aie.flow(tile_0_2, DMA, player_b_channel, tile_0_0, DMA, player_b_channel)

        @aie.core(tile_0_2)
        def core():
            for i in range_(iters):
                with aiex.hold_lock(channel_a_prod_lock, channel_a_cons_lock):
                    linalg.fill(i, output_buffer_a)
                with aiex.hold_lock(channel_b_prod_lock, channel_b_cons_lock):
                    linalg.fill(i + 1, output_buffer_b)
                yield_()

        @aie.mem(tile_0_2)
        def mem():
            aiex.send_bd(
                player_a_channel,
                channel_a_cons_lock,
                output_buffer_a,
                channel_a_prod_lock,
                repeat_count=iters - 1,
            )
            aiex.send_bd(
                player_b_channel,
                channel_b_cons_lock,
                output_buffer_b,
                channel_b_prod_lock,
                repeat_count=iters - 1,
            )
            aie.end()

        result_buffer = aie.external_buffer((K,), T.i32(), name=False)
        player_a_shim_lock = aie.lock(tile_0_0, init=0, lock_id=0, sym_name=False)
        player_b_shim_lock = aie.lock(tile_0_0, init=0, sym_name=False)

        @aie.shim_dma(tile_0_0)
        def shim():
            aiex.receive_bd(
                player_a_channel,
                player_a_shim_lock,
                result_buffer,
                player_b_shim_lock,
                repeat_count=iters - 1,
            )
            aiex.receive_bd(
                player_b_channel,
                player_b_shim_lock,
                result_buffer,
                player_a_shim_lock,
                repeat_count=iters - 1,
                offset=4,
                len=K - 8,
                dimensions=[bd_dim_layout(K // 2, 2)],
            )
            aie.end()

    compile_without_vectorization(ctx.module, workdir)
    buffer_args = ["result"]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)
    ipu_insts = aiex.ipu.get_prolog()

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)], np.int32)

        for bd_id in [0, 1]:
            ipu_insts.extend(
                aiex.ipu._update_tensor_addr_shim_tile(
                    col, bd_id, xclbin._get_buffer_host_address(0)
                )
            )
        ipu_insts.extend(aiex.ipu.lock_release(col, lock_id=0, lock_val=1))
        ipu_insts.extend(aiex.ipu.enable_cores(col, compute_tile_row))
        ipu_insts.extend(aiex.ipu.sync(column=col))

        xclbin.load_ipu_instructions(ipu_insts)

        views = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for bd_id, v in enumerate(views):
                print(f"{bd_id=}", v)


def test_double_pump_single_core_buffer(ctx: MLIRContext, workdir: Path):
    K = 32

    player_a_channel = 0
    player_b_channel = 1
    result_channel = 0
    source_channels = {}
    dest_channels = {}

    @aie.device(AIEDevice.ipu)
    def ipu():
        tiles = TileArray(cols=[0], rows=[0, 1, 2])
        buffer = aie.buffer(tiles[0, 2].tile, (K,), T.i32(), name="double_buffer")
        result_buffer = aie.buffer(
            tiles[0, 2].tile,
            (K,),
            T.i32(),
            name="result_buffer",
            initial_value=np.ones((K,), dtype=np.int32) * 3,
        )

        # available to be written to
        lock_X = tiles[0, 2].lock(init=1, sym_name="lock_X")
        # available to be read from
        # whose turn it is to write
        lock_Y = tiles[0, 2].lock(init=0, sym_name="lock_Y")

        result_prod_lock = tiles[0, 2].lock(init=1, sym_name=False)
        result_cons_lock = tiles[0, 2].lock(init=0, sym_name=False)

        # For your case I guess you could have lock X represent whether the buffer is available to be written,
        # lock Y represent whose turn it is to write. Lock X starts at 1 and lock Y starts at 0.
        # and player B does the same thing but with acq_eq(Y, 1) and rel(Y, -1) instead. (e

        @aie.mem(tiles[0, 2].tile)
        def mem():

            # Player A does acq_eq(Y, 0),zero_len,no_rel -> acq_eq(X, 1),data,rel(X, -1) -> no_acq,zero_len,rel(Y, 1)
            @aie.dma(S2MM, player_a_channel, num_blocks=3, sym_name="player_a")
            def player_a():
                # rel_val=0 means no release because these are counting semaphores and hence
                # this no change (DMA might not even issue a release)
                with aiex.hold_lock(
                    lock_Y, lock_Y, acq_action=LockAction.Acquire, acq_val=0, rel_val=0
                ):
                    aie.dma_bd(buffer, len=0)

            @aie.another_bd(player_a)
            def _():
                with aiex.hold_lock(
                    lock_X, lock_X, acq_action=LockAction.Acquire, acq_val=1, rel_val=-1
                ):
                    aie.dma_bd(buffer)

            @aie.another_bd(player_a)
            def _():
                with aiex.hold_lock(
                    lock_Y,
                    lock_Y,
                    acq_action=LockAction.Acquire,
                    acq_en=False,
                    rel_val=1,
                ):
                    aie.dma_bd(buffer)

            # Player B does acq_eq(Y, 1),zero_len,no_rel -> acq_eq(X, 1),data,rel(X, -1) -> no_acq,zero_len,rel(Y, -1)
            @aie.dma(S2MM, player_b_channel, num_blocks=3, sym_name="player_b")
            def player_b():
                # rel_val=0 means no release because these are counting semaphores and hence
                # this no change (DMA might not even issue a release)
                with aiex.hold_lock(
                    lock_Y, lock_Y, acq_action=LockAction.Acquire, acq_val=1, rel_val=0
                ):
                    aie.dma_bd(buffer, len=0)

            @aie.another_bd(player_b)
            def _():
                with aiex.hold_lock(
                    lock_X, lock_X, acq_action=LockAction.Acquire, acq_val=1, rel_val=-1
                ):
                    aie.dma_bd(buffer)

            @aie.another_bd(player_b)
            def _():
                with aiex.hold_lock(
                    lock_Y,
                    lock_Y,
                    acq_action=LockAction.Acquire,
                    acq_en=False,
                    rel_val=-1,
                ):
                    aie.dma_bd(buffer)

            aiex.send_bd(
                result_channel, result_cons_lock, result_buffer, result_prod_lock
            )

            aie.end()

        @aie.core(tiles[0, 2].tile)
        def core():
            with (
                # aiex.hold_lock(
                #     lock_X, lock_X, acq_action=LockAction.Acquire, acq_val=0, rel_val=1
                # ),
                aiex.hold_lock(result_prod_lock, result_cons_lock),
            ):
                linalg.copy(buffer, result_buffer)

        shim_to_mem_player_a_source_channel = shim_to_mem_player_a_dest_channel = (
            player_a_channel
        )
        shim_to_mem_player_b_source_channel = shim_to_mem_player_b_dest_channel = (
            player_b_channel
        )

        shim_to_mem_flow_1 = aie.flow(
            tiles[0, 0].tile,
            DMA,
            shim_to_mem_player_a_source_channel,
            tiles[0, 1].tile,
            DMA,
            shim_to_mem_player_a_dest_channel,
        )
        shim_to_mem_flow_2 = aie.flow(
            tiles[0, 0].tile,
            DMA,
            shim_to_mem_player_b_source_channel,
            tiles[0, 1].tile,
            DMA,
            shim_to_mem_player_b_dest_channel,
        )

        mem_to_core_flow_1 = aie.flow(
            tiles[0, 1].tile,
            DMA,
            player_a_channel,
            tiles[0, 2].tile,
            DMA,
            player_a_channel,
        )
        mem_to_core_flow_2 = aie.flow(
            tiles[0, 1].tile,
            DMA,
            player_b_channel,
            tiles[0, 2].tile,
            DMA,
            player_b_channel,
        )

        tile_1_0 = aie.tile(1, 0)
        core_to_shim_flow = aie.flow(
            tiles[0, 2].tile, DMA, result_channel, tile_1_0, DMA, result_channel
        )
        dest_channels["result"] = int(core_to_shim_flow.dest_channel)

        @aie.memtile_dma(tiles[0, 1].tile)
        def memtile_dma():
            buffer = aie.buffer(tiles[0, 1].tile, (K,), T.i32())
            aiex.forward_bd(tiles[0, 1].tile, buffer, player_a_channel, repeat_count=10)
            aiex.forward_bd(tiles[0, 1].tile, buffer, player_b_channel, repeat_count=10)
            aie.end()

        source_channels["player_a"] = shim_to_mem_player_a_source_channel
        source_channels["player_b"] = shim_to_mem_player_b_source_channel

    print(ctx.module)

    compile_without_vectorization(ctx.module, workdir)
    buffer_args = ["player_a", "player_b", "result"]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * len(buffer_args), np.int32)

        ipu_insts = aiex.ipu.get_prolog()
        col = 0
        for i, player in enumerate(["player_a", "player_b"]):
            bd_id = i
            source_channel = source_channels[player]
            writebd_shimtile_insts = aiex.ipu.writebd_shimtile(
                col, bd_id, buffer_length=K
            )
            ipu_insts.extend(
                aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                    writebd_shimtile_insts,
                    tensor_addr=xclbin._get_buffer_host_address(i),
                )
            )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(MM2S, source_channel, col, bd_id)
            )

        col = 1
        bd_id = 0
        dest_channel = dest_channels["result"]
        writebd_shimtile_insts = aiex.ipu.writebd_shimtile(col, bd_id, buffer_length=K)
        ipu_insts.extend(
            aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                writebd_shimtile_insts,
                tensor_addr=xclbin._get_buffer_host_address(2),
            )
        )
        ipu_insts.extend(aiex.ipu.shimtile_push_queue(S2MM, dest_channel, col, bd_id))
        ipu_insts.extend(aiex.ipu.sync(col, channel=dest_channel))

        xclbin.load_ipu_instructions(ipu_insts)

        player_a_data, player_b_data, result_data = list(map(np.asarray, views))
        print(f"{result_data=}")
        A = np.random.randint(0, 10, (K,), dtype=np.int32)
        B = np.random.randint(0, 10, (K,), dtype=np.int32)

        np.copyto(player_a_data, A, casting="no")
        np.copyto(player_b_data, B, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            print(result_data)
