# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.


from pathlib import Path
import sys

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
)

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref, scf

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
import aie.extras.types as T
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

range_ = scf.range_
yield_ = scf.yield_


def test_manual_args(ctx: MLIRContext, workdir: Path):
    K = 32
    RANDOM_WEIGHT = np.random.randint(0, 10, (K,), dtype=np.int32)
    iters = 10
    loop = False

    @aie.device(AIEDevice.npu)
    def npu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        weight = memref.global_(initial_value=RANDOM_WEIGHT, constant=True)
        buffer_weight = aie.buffer(tile_0_2, (K,), T.i32())
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        @aie.core(tile_0_2)
        def core():
            x = memref.get_global(weight.type_.value, weight.sym_name.value)
            y = memref.alloc(K, T.i32())
            for j in range(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.fill(j, y)
                    linalg.copy(x, buffer_weight)
                    linalg.add(y, buffer_weight, buffer_weight)

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

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir)
    buffer_args = [f"out_{i}" for i in range(iters)]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * iters, np.int32)

        col = 0
        channel_index = 0
        npu_insts = aiex.npu.get_prolog()
        for bd_id in range(iters):
            writebd_shimtile_insts = aiex.npu.writebd_shimtile(
                col, bd_id, buffer_length=K
            )
            npu_insts.extend(
                aiex.npu._exec_write_bd_extend_shim_tile_opt(
                    writebd_shimtile_insts,
                    tensor_addr=xclbin._get_buffer_host_address(bd_id),
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            npu_insts.extend(aiex.npu.sync(column=col))

        xclbin.load_npu_instructions(npu_insts)

        wraps = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        for i, w in enumerate(wraps):
            if not np.array_equal(RANDOM_WEIGHT + i, w):
                with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                    print("RANDOM_WEIGHT", RANDOM_WEIGHT)
                    print(f"{buffer_args[i]} =", w)
                    assert False


def test_manual_args_with_offset(ctx: MLIRContext, workdir: Path):
    K = 32
    RANDOM_WEIGHT = np.random.randint(0, 10, (K,), dtype=np.int32)
    iters = 10
    loop = False

    @aie.device(AIEDevice.npu)
    def npu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        weight = memref.global_(initial_value=RANDOM_WEIGHT, constant=True)
        buffer_weight = aie.buffer(tile_0_2, (K,), T.i32())
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        @aie.core(tile_0_2)
        def core():
            x = memref.get_global(weight.type_.value, weight.sym_name.value)
            y = memref.alloc(K, T.i32())
            for j in range(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.fill(j, y)
                    linalg.copy(x, buffer_weight)
                    linalg.add(y, buffer_weight, buffer_weight)

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

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir)
    buffer_args = [f"out_{i}" for i in range(iters)]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K * iters,)] * iters, np.int32)

        col = 0
        channel_index = 0
        npu_insts = aiex.npu.get_prolog()
        for i in range(iters):
            bd_id = i
            writebd_shimtile_insts = aiex.npu.writebd_shimtile(
                col, bd_id, buffer_length=K, buffer_offset=K * i
            )
            npu_insts.extend(
                aiex.npu._exec_write_bd_extend_shim_tile_opt(
                    writebd_shimtile_insts,
                    tensor_addr=xclbin._get_buffer_host_address(i),
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            npu_insts.extend(aiex.npu.sync(column=col))

        xclbin.load_npu_instructions(npu_insts)

        wraps = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        for i, w in enumerate(wraps):
            if not np.array_equal(RANDOM_WEIGHT + i, w.reshape((iters, K))[i]):
                with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                    print("RANDOM_WEIGHT", RANDOM_WEIGHT + i)
                    print(f"{buffer_args[i]} =", w.reshape((iters, K))[i])
                    assert False


def test_manual_args_with_different_cols(ctx: MLIRContext, workdir: Path):
    K = 32
    RANDOM_WEIGHT = np.random.randint(0, 10, (K,), dtype=np.int32)
    cols = [0, 1, 2, 3]

    @aie.device(AIEDevice.npu)
    def npu():
        for c in cols:
            tile_c_0 = aie.tile(c, 0)
            tile_c_2 = aie.tile(c, 2)

            buffer_weight = aie.buffer(
                tile_c_2, (K,), T.i32(), initial_value=RANDOM_WEIGHT
            )
            lock_read_weight = aie.lock(tile_c_2, init=1)
            lock_send_weight = aie.lock(tile_c_2, init=0)

            aie.flow(tile_c_2, DMA, 0, tile_c_0, DMA, 0)

            @aie.core(tile_c_2)
            def core():
                y = memref.alloc(K, T.i32())
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.fill(c, y)
                    linalg.add(y, buffer_weight, buffer_weight)

            @aie.mem(tile_c_2)
            def mem_c_2():
                @aie.dma(MM2S, 0)
                def dma3():
                    aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                    aie.dma_bd(buffer_weight)
                    aie.use_lock(lock_read_weight, Release)

                aie.end()

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir)
    buffer_args = [f"out_{c}" for c in cols]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * len(cols), np.int32)

        bd_id = 0
        channel_index = 0
        npu_insts = aiex.npu.get_prolog()
        for col in cols:
            writebd_shimtile_insts = aiex.npu.writebd_shimtile(
                col, bd_id, buffer_length=K
            )
            npu_insts.extend(
                aiex.npu._exec_write_bd_extend_shim_tile_opt(
                    writebd_shimtile_insts,
                    tensor_addr=xclbin._get_buffer_host_address(col),
                )
            )
            npu_insts.extend(
                aiex.npu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            npu_insts.extend(aiex.npu.sync(column=col))

        xclbin.load_npu_instructions(npu_insts)

        wraps = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        for c, w in enumerate(wraps):
            if not np.array_equal(RANDOM_WEIGHT + c, w):
                with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                    print("RANDOM_WEIGHT", RANDOM_WEIGHT + c)
                    print(f"{buffer_args[c]} =", w)
                    assert False


def test_manual_args_with_shim_dma(ctx: MLIRContext, workdir: Path):
    K = 32
    cols = [2]
    compute_tile_row = 2

    iters = 21

    @aie.device(AIEDevice.npu)
    def npu():
        if 0 not in cols:
            tile_dummy = aie.tile(0, 3)
        for c in cols:
            tile_c_0 = aie.tile(c, 0)
            tile_c_2 = aie.tile(c, compute_tile_row)

            buffer_weight = aie.buffer(
                tile_c_2, (K,), T.i32(), initial_value=np.ones((K,), dtype=np.int32) * c
            )
            lock_read_weight = aie.lock(tile_c_2, init=1)
            lock_send_weight = aie.lock(tile_c_2, init=0)

            aie.flow(tile_c_2, DMA, 0, tile_c_0, DMA, 0)

            @aie.core(tile_c_2)
            def core():
                y = memref.alloc(K, T.i32())
                for i in range_(iters):
                    with aiex.hold_lock(lock_read_weight, lock_send_weight):
                        linalg.fill(i, y)
                        linalg.copy(y, buffer_weight)
                    yield_()

            @aie.mem(tile_c_2)
            def mem_c_2():
                @aie.dma(MM2S, 0, loop=False, repeat_count=iters - 1)
                def dma3():
                    aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                    aie.dma_bd(buffer_weight)
                    aie.use_lock(lock_read_weight, Release)

                aie.end()

            lock_c_0_read_in_c = aie.lock(tile_c_0, init=0, lock_id=0)
            host_buffer = aie.external_buffer((K,), T.i32())

            @aie.shim_dma(tile_c_0)
            def shim_dma_c_0():
                @aie.dma(S2MM, 0, loop=False, repeat_count=iters - 1)
                def dma():
                    aie.use_lock(lock_c_0_read_in_c, Acquire, acq_en=0, value=0)
                    aie.dma_bd(host_buffer)
                    aie.use_lock(lock_c_0_read_in_c, Release, value=0)

                aie.end()

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir, enable_cores=False)
    buffer_args = [f"out_{c}" for c in cols]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * len(cols), np.int32)

        bd_id = 0
        npu_insts = aiex.npu.get_prolog()
        for i, col in enumerate(cols):
            update_addrs = aiex.npu._update_tensor_addr_shim_tile(
                col, bd_id, tensor_addr=xclbin._get_buffer_host_address(i)
            )
            npu_insts.extend(update_addrs)
            npu_insts.extend(aiex.npu.enable_cores(col, compute_tile_row))

        xclbin.load_npu_instructions(npu_insts)

        wraps = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        print(f"{iters=}")
        for col, w in zip(cols, wraps):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(f"{col=}")
                print(w)
