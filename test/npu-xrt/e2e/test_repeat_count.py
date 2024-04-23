# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import random
import sys

from aie.dialects import aie, aiex, scf
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)

range_ = scf.for_
yield_ = scf.yield_

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref
import aie.extras.types as T
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np


from pathlib import Path
import pytest

# noinspection PyUnresolvedReferences
from aie.extras.testing import mlir_ctx as ctx, filecheck, MLIRContext

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")

from aie.compiler.util import (
    compile_without_vectorization,
    make_xclbin,
)

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def test_repeat_count(ctx: MLIRContext, workdir: Path):
    K = 32
    iters = 4
    loop = False
    RANDOM_WEIGHT = np.random.randint(0, 10, (K,), dtype=np.int32)
    npu_insts = aiex.npu.get_prolog()

    @aie.device(AIEDevice.npu)
    def npu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        buffer_weight = aie.buffer(tile_0_2, (K,), T.i32(), initial_value=RANDOM_WEIGHT)
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        @aie.core(tile_0_2)
        def core():
            for _ in range(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.copy(buffer_weight, buffer_weight)

        @aie.mem(tile_0_2)
        def mem_0_2():
            @aie.dma(MM2S, 0, loop=loop, repeat_count=iters - 1)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            lock_0_1_read_in_c = aie.lock(tile_0_1, init=1)
            lock_0_1_write_out_c = aie.lock(tile_0_1, init=0)
            buffer_0_1_c = aie.buffer(tile_0_1, (K,), T.i32())

            @aie.dma(S2MM, 2, loop=loop, repeat_count=iters - 1)
            def dma5():
                aie.use_lock(lock_0_1_read_in_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_write_out_c, Release)

            @aie.dma(MM2S, 2, loop=loop, repeat_count=iters - 1)
            def dma6():
                aie.use_lock(lock_0_1_write_out_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_read_in_c, Release)

            aie.end()

        # in A
        channel_index = 0
        ddr_id = 0
        col = 0
        for i, bd_id in enumerate(range(iters)):
            npu_insts.extend(
                aiex.npu.writebd_shimtile(
                    col,
                    bd_id,
                    buffer_length=K,
                    buffer_offset=i * K,
                    ddr_id=ddr_id,
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            npu_insts.extend(
                aiex.npu.sync(
                    channel=0,
                    column=col,
                    column_num=1,
                    direction=0,
                    row=0,
                    row_num=1,
                )
            )

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_npu_instructions(npu_insts)
        views = xclbin.mmap_buffers([(iters * K,)], np.int32)

        wrap_C = np.asarray(views[0])

        C = np.zeros((iters * K,), dtype=np.int32)
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(np.concatenate([RANDOM_WEIGHT] * iters), wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(RANDOM_WEIGHT)
                print(wrap_C)
                assert False


def test_no_loop(ctx: MLIRContext, workdir: Path):
    K = 32
    # RANDOM_WEIGHT = np.random.randint(0, 10, (K,), dtype=np.int32)
    RANDOM_WEIGHT = np.ones((K,), dtype=np.int32) * random.randint(1, 100)
    col = 2
    iters = 10
    npu_insts = aiex.npu.get_prolog()

    @aie.device(AIEDevice.npu)
    def npu():
        nonlocal col

        if col != 0:
            tile_dummy = aie.tile(0, 3)
        tile_c_0 = aie.tile(col, 0)
        tile_c_2 = aie.tile(col, 2)

        buffer_weight = aie.buffer(tile_c_2, (K,), T.i32(), initial_value=RANDOM_WEIGHT)
        lock_read_weight = aie.lock(tile_c_2, init=1)
        lock_send_weight = aie.lock(tile_c_2, init=0)

        aie.flow(tile_c_2, DMA, 0, tile_c_0, DMA, 0)

        @aie.core(tile_c_2)
        def core():
            y = memref.alloc(K, T.i32())
            for i in range_(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.fill(col, y)
                    linalg.add(y, buffer_weight, buffer_weight)
                yield_([])

        @aie.mem(tile_c_2)
        def mem_c_2():
            @aie.dma(MM2S, 0, loop=False, repeat_count=iters - 1)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)

    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)], np.int32)

        channel_index = 0
        ddr_id = 0
        bd_id = 0
        npu_insts.extend(
            aiex.npu.writebd_shimtile(
                col,
                bd_id,
                buffer_length=K,
                ddr_id=ddr_id,
            )
        )
        npu_insts.extend(
            aiex.npu.shimtile_push_queue(
                S2MM, channel_index, col, bd_id, repeats=iters - 1
            )
        )
        npu_insts.extend(
            aiex.npu.sync(
                channel=0,
                column=col,
                column_num=1,
                direction=S2MM,
                row=0,
                row_num=1,
            )
        )

        xclbin.load_npu_instructions(npu_insts)

        wraps = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(RANDOM_WEIGHT + (col * iters), wraps[0]):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(f"{RANDOM_WEIGHT + (col * iters)=}")
                print(f"{wraps[0]=}")

                assert False
