# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
from pathlib import Path
import random
import sys
import time

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref

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
from aie.dialects import aie, aiex, scf
from aie.dialects.aie import AIEDevice, DMAChannelDir, LockAction, WireBundle
import aie.extras.types as T
from aie.xrt import XCLBin

range_ = scf.for_
yield_ = scf.yield_

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")

pytest.mark.usefixtures("run_around_tests")

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


def test_one_global(ctx: MLIRContext, workdir: Path):
    K = 32
    RANDOM_NUMBER = random.randint(0, 100)
    iv = np.random.randint(0, 10, (K,), dtype=np.int32)
    column = 2

    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        # TODO(max): figure this annoying thing out...
        if column != 0:
            _dummy_tile = aie.tile(0, 2)

        tile_2_3 = aie.tile(column, 3)
        global_weight_2_3 = memref.global_(initial_value=iv)
        # if you stick the locks inside the core then it gets dce'd???
        lock_weight_2_3 = aie.lock(tile_2_3, init=0)

        @aie.core(tile_2_3)
        def core_2_3():
            with aiex.hold_lock(lock_weight_2_3, lock_weight_2_3, acq_val=0):
                x = memref.get_global(
                    global_weight_2_3.type_.value, global_weight_2_3.sym_name.value
                )
                # this doesn't actually do anything...?
                # right now just functions as a way to not DCE
                linalg.fill(RANDOM_NUMBER, x)

        tile_2_2 = aie.tile(column, 2)
        # if you stick the locks inside the core then it gets dce'd???
        lock_use_weight_2_2 = aie.lock(tile_2_2, init=0)
        # if you stick a buffer into the core then it doesn't get injected into the elf???
        buffer_weight_2_2 = aie.buffer(tile_2_2, (K,), T.i32())

        @aie.core(tile_2_2)
        def core_2_2():
            with aiex.hold_lock(lock_weight_2_3, lock_use_weight_2_2):
                x = memref.get_global(
                    global_weight_2_3.type_.value, global_weight_2_3.sym_name.value
                )
                linalg.copy(x, buffer_weight_2_2)

        mem_tile = aie.tile(column, 1)
        flow_to_mem = aie.flow(tile_2_2, dest=mem_tile)

        @aie.mem(tile_2_2)
        def mem():
            @aie.dma(
                MM2S,
                flow_to_mem.source_channel,
            )
            def _():
                aiex.process_bd(
                    lock_use_weight_2_2, buffer_weight_2_2, lock_use_weight_2_2
                )

            aie.end()

        shim_tile = aie.tile(column, 0)
        flow_to_shim = aie.flow(mem_tile, dest=shim_tile)
        mem_tile_buffer = aie.buffer(mem_tile, (K,), T.i32())

        @aie.memtile_dma(mem_tile)
        def memtile_dma():
            aiex.forward_bd(
                mem_tile,
                mem_tile_buffer,
                s2mm_channel_idx=flow_to_mem.dest_channel,
                mm2s_channel_idx=flow_to_shim.source_channel,
            )

            aie.end()

        ddr_id = 0
        bd_id = 0
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                column=column,
                bd_id=bd_id,
                length=K,
                buffer_offset=0,
                ddr_id=ddr_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                channel_dir=S2MM,
                channel_index=flow_to_shim.dest_channel,
                column=column,
                bd_id=bd_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=flow_to_shim.dest_channel,
                column=column,
                direction=0,
                row=0,
            )
        )

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        [c] = xclbin.mmap_buffers([(K,)], np.int32)
        wrap_C = np.asarray(c)
        C = np.zeros((K,), dtype=np.int32)
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(iv, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(f"{iv=}")
                print(f"c={wrap_C}")
                assert False


def test_threesome(ctx: MLIRContext, workdir: Path):
    K = 32
    iv1 = np.random.randint(0, 10, (K,), dtype=np.int32)
    iv2 = np.random.randint(0, 10, (K,), dtype=np.int32)

    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        _dummy_tile = aie.tile(0, 2)
        tile_1_2 = aie.tile(1, 2)
        global_weight_1_2 = memref.global_(initial_value=iv1)
        lock_weight_1_2 = aie.lock(tile_1_2, init=0)

        @aie.core(tile_1_2)
        def core_1_2():
            with aiex.hold_lock(lock_weight_1_2, lock_weight_1_2, acq_val=0):
                x = memref.get_global(
                    global_weight_1_2.type_.value, global_weight_1_2.sym_name.value
                )
                linalg.fill(0, x)

        tile_2_3 = aie.tile(2, 3)
        global_weight_2_3 = memref.global_(initial_value=iv2)
        lock_weight_2_3 = aie.lock(tile_2_3, init=0)

        @aie.core(tile_2_3)
        def core_2_3():
            with aiex.hold_lock(lock_weight_2_3, lock_weight_2_3, acq_val=0):
                x = memref.get_global(
                    global_weight_2_3.type_.value, global_weight_2_3.sym_name.value
                )
                linalg.fill(0, x)

        tile_2_2 = aie.tile(2, 2)
        lock_use_weight_2_2 = aie.lock(tile_2_2, init=0)
        buffer_weight_2_2 = aie.buffer(tile_2_2, (K,), T.i32())

        @aie.core(tile_2_2)
        def core_2_2():
            with (
                aiex.hold_lock(lock_weight_1_2, lock_weight_1_2),
                aiex.hold_lock(lock_weight_2_3, lock_weight_2_3),
                aiex.hold_lock(lock_use_weight_2_2, lock_use_weight_2_2, acq_val=0),
            ):
                x = memref.get_global(
                    global_weight_1_2.type_.value, global_weight_1_2.sym_name.value
                )
                linalg.copy(x, buffer_weight_2_2)
                y = memref.get_global(
                    global_weight_2_3.type_.value, global_weight_2_3.sym_name.value
                )
                linalg.add(y, buffer_weight_2_2, buffer_weight_2_2)

        shim_tile_column = 2
        mem_tile = aie.tile(shim_tile_column, 1)
        flow_to_mem = aie.flow(tile_2_2, dest=mem_tile)

        @aie.mem(tile_2_2)
        def mem_2_2():
            @aie.dma(
                MM2S,
                flow_to_mem.source_channel,
            )
            def _():
                aiex.process_bd(
                    lock_use_weight_2_2, buffer_weight_2_2, lock_use_weight_2_2
                )

            aie.end()

        shim_tile = aie.tile(shim_tile_column, 0)
        flow_to_shim = aie.flow(mem_tile, dest=shim_tile)
        mem_tile_buffer = aie.buffer(mem_tile, (K,), T.i32())

        @aie.memtile_dma(mem_tile)
        def memtile_dma():
            aiex.forward_bd(
                mem_tile,
                mem_tile_buffer,
                s2mm_channel_idx=flow_to_mem.dest_channel,
                mm2s_channel_idx=flow_to_shim.source_channel,
            )

            aie.end()

        ddr_id = 0
        bd_id = 0
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                column=shim_tile_column,
                bd_id=bd_id,
                length=K,
                buffer_offset=0,
                ddr_id=ddr_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                channel_dir=S2MM,
                channel_index=flow_to_shim.dest_channel,
                column=shim_tile_column,
                bd_id=bd_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=flow_to_shim.dest_channel,
                column=shim_tile_column,
                direction=0,
                row=0,
            )
        )

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        [c] = xclbin.mmap_buffers([(K,)], np.int32)
        wrap_C = np.asarray(c)
        C = np.zeros((K,), dtype=np.int32)
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(iv1 + iv2, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(f"{iv1=}")
                print(f"{iv2=}")
                print(f"c={wrap_C}")
                assert False


def test_foursome(ctx: MLIRContext, workdir: Path):
    K = 32
    iv1 = np.random.randint(0, 10, (K,), dtype=np.int32)
    iv2 = np.random.randint(0, 10, (K,), dtype=np.int32)
    iv3 = np.random.randint(0, 10, (K,), dtype=np.int32)

    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        _dummy_tile = aie.tile(0, 2)

        tile_1_3 = aie.tile(1, 3)
        global_weight_1_3 = memref.global_(initial_value=iv1)
        lock_weight_1_3 = aie.lock(tile_1_3, init=0)

        @aie.core(tile_1_3)
        def core():
            with aiex.hold_lock(lock_weight_1_3, lock_weight_1_3, acq_val=0):
                x = memref.get_global(
                    global_weight_1_3.type_.value, global_weight_1_3.sym_name.value
                )
                linalg.fill(0, x)

        tile_2_4 = aie.tile(2, 4)
        global_weight_2_4 = memref.global_(initial_value=iv2)
        lock_weight_2_4 = aie.lock(tile_2_4, init=0)

        @aie.core(tile_2_4)
        def core():
            with aiex.hold_lock(lock_weight_2_4, lock_weight_2_4, acq_val=0):
                x = memref.get_global(
                    global_weight_2_4.type_.value, global_weight_2_4.sym_name.value
                )
                linalg.fill(0, x)

        tile_2_2 = aie.tile(2, 2)
        global_weight_2_2 = memref.global_(initial_value=iv3)
        lock_weight_2_2 = aie.lock(tile_2_2, init=0)

        @aie.core(tile_2_2)
        def core():
            with aiex.hold_lock(lock_weight_2_2, lock_weight_2_2, acq_val=0):
                x = memref.get_global(
                    global_weight_2_2.type_.value, global_weight_2_2.sym_name.value
                )
                linalg.fill(0, x)

        tile_2_3 = aie.tile(2, 3)
        lock_use_weight_2_3 = aie.lock(tile_2_3, init=0)
        buffer_weight_2_3 = aie.buffer(tile_2_3, (K,), T.i32())

        @aie.core(tile_2_3)
        def core():
            with (
                aiex.hold_lock(lock_weight_1_3, lock_weight_1_3),
                aiex.hold_lock(lock_weight_2_2, lock_weight_2_2),
                aiex.hold_lock(lock_weight_2_4, lock_weight_2_4),
                aiex.hold_lock(lock_use_weight_2_3, lock_use_weight_2_3, acq_val=0),
            ):
                x = memref.get_global(
                    global_weight_1_3.type_.value, global_weight_1_3.sym_name.value
                )
                linalg.copy(x, buffer_weight_2_3)
                y = memref.get_global(
                    global_weight_2_4.type_.value, global_weight_2_4.sym_name.value
                )
                linalg.add(y, buffer_weight_2_3, buffer_weight_2_3)
                z = memref.get_global(
                    global_weight_2_2.type_.value, global_weight_2_2.sym_name.value
                )
                linalg.add(z, buffer_weight_2_3, buffer_weight_2_3)

        shim_tile_column = 3
        mem_tile = aie.tile(shim_tile_column, 1)
        aie.flow(tile_2_3, dest=mem_tile)
        flow_to_mem = aie.flow(tile_2_3, dest=mem_tile)

        @aie.mem(tile_2_3)
        def mem():
            @aie.dma(
                MM2S,
                flow_to_mem.source_channel,
            )
            def _():
                aiex.process_bd(
                    lock_use_weight_2_3, buffer_weight_2_3, lock_use_weight_2_3
                )

            aie.end()

        shim_tile = aie.tile(shim_tile_column, 0)
        flow_to_shim = aie.flow(mem_tile, dest=shim_tile)
        mem_tile_buffer = aie.buffer(mem_tile, (K,), T.i32())

        @aie.memtile_dma(mem_tile)
        def memtile_dma():
            aiex.forward_bd(
                mem_tile,
                mem_tile_buffer,
                s2mm_channel_idx=flow_to_mem.dest_channel,
                mm2s_channel_idx=flow_to_shim.source_channel,
            )

            aie.end()

        ddr_id = 0
        bd_id = 0
        ipu_insts.extend(
            aiex.ipu.writebd_shimtile(
                column=shim_tile_column,
                bd_id=bd_id,
                length=K,
                buffer_offset=0,
                ddr_id=ddr_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.shimtile_push_queue(
                channel_dir=S2MM,
                channel_index=flow_to_shim.dest_channel,
                column=shim_tile_column,
                bd_id=bd_id,
            )
        )
        ipu_insts.extend(
            aiex.ipu.sync(
                channel=flow_to_shim.dest_channel,
                column=shim_tile_column,
                direction=0,
                row=0,
            )
        )

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        [c] = xclbin.mmap_buffers([(K,)], np.int32)
        wrap_C = np.asarray(c)
        C = np.zeros((K,), dtype=np.int32)
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(iv1 + iv2 + iv3, wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(f"{iv1=}")
                print(f"{iv2=}")
                print(f"{iv3=}")
                print(f"c={wrap_C}")
                assert False


def test_single_prod_mult_cons_with_sync_bd(ctx: MLIRContext, workdir: Path):
    K = 32
    iters = 20
    ipu_insts = aiex.ipu.get_prolog()

    col_shim_channel_index = {}

    N_CONSUMERS = 3

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        tile_1_0 = aie.tile(1, 0)
        tile_2_0 = aie.tile(2, 0)

        buffer_weight = aie.buffer(tile_0_2, (K,), T.i32())
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        core_to_mem_fl = aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)

        @aie.core(tile_0_2)
        def core():
            for i in range_(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.fill(i + 1, buffer_weight)
                yield_([])

        @aie.mem(tile_0_2)
        def mem_0_2():
            @aie.dma(MM2S, core_to_mem_fl.source_channel, repeat_count=iters - 1)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

        # try to use different channels as much as possible to prevent false positives
        mem_to_shim_tile_0_0_fl = aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)
        col_shim_channel_index[0] = int(mem_to_shim_tile_0_0_fl.dest_channel)

        mem_to_shim_tile_1_0_fl = aie.flow(tile_0_1, DMA, 2, tile_1_0, DMA, 1)
        col_shim_channel_index[1] = int(mem_to_shim_tile_1_0_fl.dest_channel)

        mem_to_shim_tile_2_0_fl = aie.flow(tile_0_1, DMA, 3, tile_2_0, DMA, 0)
        col_shim_channel_index[2] = int(mem_to_shim_tile_2_0_fl.dest_channel)

        consumer_flows = [
            mem_to_shim_tile_0_0_fl,
            mem_to_shim_tile_1_0_fl,
            mem_to_shim_tile_2_0_fl,
        ]

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            buffer_0_1_c = aie.buffer(tile_0_1, (K,), T.i32())

            lock_0_1_read_in_c = aie.lock(tile_0_1, init=N_CONSUMERS)
            lock_0_1_write_out_c = aie.lock(tile_0_1, init=0)

            @aie.dma(S2MM, core_to_mem_fl.dest_channel, repeat_count=iters - 1)
            def dma5():
                aie.use_lock(lock_0_1_read_in_c, AcquireGreaterEqual, value=N_CONSUMERS)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_write_out_c, Release, value=2 * N_CONSUMERS)

            for fl in consumer_flows:

                @aie.dma(MM2S, fl.source_channel, repeat_count=iters - 1, num_bds=2)
                def dma6():
                    aie.use_lock(lock_0_1_write_out_c, AcquireGreaterEqual, value=2)
                    aie.dma_bd(buffer_0_1_c)
                    # don't release any lock (no-op)
                    aie.use_lock(lock_0_1_read_in_c, Release, value=0)

                @aie.another_bd(dma6)
                def sync_bd():
                    # acquire after all consumers have started (ie when the sema is down to 0 after all have decremented)
                    aie.use_lock(lock_0_1_write_out_c, Acquire, value=0)
                    aie.dma_bd(buffer_0_1_c, len=0)
                    aie.use_lock(lock_0_1_read_in_c, Release, value=1)

            aie.end()

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir)

    buffer_args = {f"receive_a_col_{i}": (iters * K,) for i in range(N_CONSUMERS)}
    kernel_json = emit_design_kernel_json(buffer_args=list(buffer_args.keys()))
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(list(buffer_args.values()), np.int32)
        for col, shim_channel_idx in col_shim_channel_index.items():
            bd_id = 0
            ipu_insts.extend(
                aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                    aiex.ipu.writebd_shimtile(
                        col,
                        bd_id=bd_id,
                        length=K,
                        iteration_size=iters - 1,
                        iteration_stride=K,
                    ),
                    tensor_addr=xclbin._get_buffer_host_address(buffer_idx=col),
                )
            )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(
                    channel_dir=S2MM,
                    channel_index=shim_channel_idx,
                    column=col,
                    bd_id=bd_id,
                    repeats=iters - 1,
                )
            )

        for col, shim_channel_idx in col_shim_channel_index.items():
            ipu_insts.extend(aiex.ipu.sync(channel=shim_channel_idx, column=col))

        xclbin.load_ipu_instructions(ipu_insts)

        wrapped = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        correct = np.arange(1, iters + 1).repeat(K).reshape(iters, K).flatten()

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for w in wrapped:
                if not np.array_equal(w, correct):
                    print(w)
                    assert False


@pytest.mark.xfail(reason="should hang")
def test_single_prod_mult_cons_with_wrong_sync_bd_deadlocks(
    ctx: MLIRContext, workdir: Path
):
    K = 48
    iters = 20
    ipu_insts = aiex.ipu.get_prolog()

    col_shim_channel_index = {}

    N_CONSUMERS = 3

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        tile_1_0 = aie.tile(1, 0)
        tile_2_0 = aie.tile(2, 0)

        buffer_weight = aie.buffer(tile_0_2, (K,), T.i32())
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        core_to_mem_fl = aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)

        @aie.core(tile_0_2)
        def core():
            for i in range_(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.fill(i + 1, buffer_weight)
                yield_([])

        @aie.mem(tile_0_2)
        def mem_0_2():
            @aie.dma(MM2S, core_to_mem_fl.source_channel, repeat_count=iters - 1)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

        # try to use different channels as much as possible to prevent false positives
        mem_to_shim_tile_0_0_fl = aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)
        col_shim_channel_index[0] = int(mem_to_shim_tile_0_0_fl.dest_channel)

        mem_to_shim_tile_1_0_fl = aie.flow(tile_0_1, DMA, 2, tile_1_0, DMA, 1)
        col_shim_channel_index[1] = int(mem_to_shim_tile_1_0_fl.dest_channel)

        mem_to_shim_tile_2_0_fl = aie.flow(tile_0_1, DMA, 3, tile_2_0, DMA, 0)
        col_shim_channel_index[2] = int(mem_to_shim_tile_2_0_fl.dest_channel)

        consumer_flows = [
            mem_to_shim_tile_0_0_fl,
            mem_to_shim_tile_1_0_fl,
            mem_to_shim_tile_2_0_fl,
        ]

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            buffer_0_1_c = aie.buffer(tile_0_1, (K,), T.i32())

            lock_0_1_read_in_c = aie.lock(tile_0_1, init=N_CONSUMERS)
            lock_0_1_write_out_c = aie.lock(tile_0_1, init=0)

            @aie.dma(S2MM, core_to_mem_fl.dest_channel, repeat_count=iters - 1)
            def dma5():
                aie.use_lock(lock_0_1_read_in_c, AcquireGreaterEqual, value=N_CONSUMERS)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_write_out_c, Release, value=2 * N_CONSUMERS)

            for fl in consumer_flows:

                @aie.dma(MM2S, fl.source_channel, repeat_count=iters - 1, num_bds=2)
                def dma6():
                    aie.use_lock(lock_0_1_write_out_c, AcquireGreaterEqual, value=2)
                    aie.dma_bd(buffer_0_1_c)
                    # don't release lock_0_1_write_out_c
                    aie.use_lock(lock_0_1_read_in_c, Release, value=0)

                @aie.another_bd(dma6)
                def sync_bd():
                    # don't need to acquire lock_0_1_write_out_c since it wasn't released
                    aie.use_lock(lock_0_1_write_out_c, acq_en=False)
                    aie.dma_bd(buffer_0_1_c, len=0)
                    aie.use_lock(lock_0_1_read_in_c, Release, value=1)

            aie.end()

    assert ctx.module.operation.verify()

    compile_without_vectorization(ctx.module, workdir)

    buffer_args = {f"receive_a_col_{i}": (iters * K,) for i in range(N_CONSUMERS)}
    kernel_json = emit_design_kernel_json(buffer_args=list(buffer_args.keys()))
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(list(buffer_args.values()), np.int32)
        for col, shim_channel_idx in col_shim_channel_index.items():
            bd_id = 0
            if col == 2:
                for _ in range(10):
                    ipu_insts.extend(
                        aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                            aiex.ipu.writebd_shimtile(
                                col,
                                bd_id=bd_id,
                                length=K,
                                iteration_size=iters - 1,
                                iteration_stride=K,
                            ),
                            tensor_addr=xclbin._get_buffer_host_address(buffer_idx=col),
                        )
                    )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(
                    channel_dir=S2MM,
                    channel_index=shim_channel_idx,
                    column=col,
                    bd_id=bd_id,
                    repeats=iters - 1,
                )
            )

        for col, shim_channel_idx in col_shim_channel_index.items():
            ipu_insts.extend(aiex.ipu.sync(channel=shim_channel_idx, column=col))

        xclbin.load_ipu_instructions(ipu_insts)

        wrapped = list(map(np.asarray, views))

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        correct = np.arange(1, iters + 1).repeat(K).reshape(iters, K).flatten()

        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for w in wrapped:
                if not np.array_equal(w, correct):
                    print(w)
                    assert False


# @pytest.mark.parametrize("K", (2**i for i in range(4, 10)))
def test_single_prod_mult_cons_with_multiple_sync_bds(
    ctx: MLIRContext, workdir: Path, K=4
):
    iters = 8
    ipu_insts = aiex.ipu.get_prolog()

    col_shim_channel_index = {}

    N_CONSUMERS = 4
    N_WRITE_OUTS = 4

    dtype = T.i32()
    np_dtype = np.int32
    byte_width_dtype = dtype.width // 8

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        tile_1_0 = aie.tile(1, 0)
        tile_2_0 = aie.tile(2, 0)
        tile_3_0 = aie.tile(3, 0)

        buffer_weight = aie.buffer(tile_0_2, (N_CONSUMERS * N_WRITE_OUTS * K,), T.i32())
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        core_to_mem_fl = aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)

        @aie.core(tile_0_2)
        def core():
            for i in range_(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    for j in range(N_CONSUMERS):
                        ii = i + 1
                        jj = j + 1
                        linalg.fill(
                            ii + jj + ii * jj,
                            buffer_weight[
                                j * N_WRITE_OUTS * K : (j + 1) * N_WRITE_OUTS * K
                            ],
                        )
                yield_([])

        @aie.mem(tile_0_2)
        def mem_0_2():
            @aie.dma(MM2S, core_to_mem_fl.source_channel, repeat_count=iters - 1)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

        # try to use different channels as much as possible to prevent false positives
        mem_to_shim_tile_0_0_fl = aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)
        col_shim_channel_index[0] = int(mem_to_shim_tile_0_0_fl.dest_channel)

        mem_to_shim_tile_1_0_fl = aie.flow(tile_0_1, DMA, 2, tile_1_0, DMA, 1)
        col_shim_channel_index[1] = int(mem_to_shim_tile_1_0_fl.dest_channel)

        mem_to_shim_tile_2_0_fl = aie.flow(tile_0_1, DMA, 3, tile_2_0, DMA, 0)
        col_shim_channel_index[2] = int(mem_to_shim_tile_2_0_fl.dest_channel)

        mem_to_shim_tile_3_0_fl = aie.flow(tile_0_1, DMA, 4, tile_3_0, DMA, 0)
        col_shim_channel_index[3] = int(mem_to_shim_tile_3_0_fl.dest_channel)

        consumer_flows = [
            mem_to_shim_tile_0_0_fl,
            mem_to_shim_tile_1_0_fl,
            mem_to_shim_tile_2_0_fl,
            mem_to_shim_tile_3_0_fl,
        ]

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            buffer_0_1_c = aie.buffer(
                tile_0_1, (N_CONSUMERS * N_WRITE_OUTS * K,), dtype
            )

            read_in_locks = [
                aie.lock(tile_0_1, init=N_WRITE_OUTS, sym_name=f"read_in_{i}")
                for i in range(N_CONSUMERS)
            ]
            write_out_locks = [
                aie.lock(tile_0_1, init=0, sym_name=f"write_out_{i}")
                for i in range(N_CONSUMERS)
            ]

            # read in
            @aie.dma(
                S2MM,
                core_to_mem_fl.dest_channel,
                repeat_count=iters - 1,
                num_bds=N_CONSUMERS,
            )
            def dma5():
                aie.use_lock(read_in_locks[0], AcquireGreaterEqual, value=N_WRITE_OUTS)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(write_out_locks[0], Release, value=N_WRITE_OUTS)

            for i in range(1, N_CONSUMERS):

                @aie.another_bd(dma5)
                def _():
                    aie.use_lock(
                        read_in_locks[i], AcquireGreaterEqual, value=N_WRITE_OUTS
                    )
                    aie.dma_bd(buffer_0_1_c, len=0)
                    aie.use_lock(write_out_locks[i], Release, value=N_WRITE_OUTS)

            # write out
            for i, fl in enumerate(consumer_flows):

                @aie.dma(
                    MM2S, fl.source_channel, repeat_count=(iters * N_WRITE_OUTS) - 1
                )
                def dma6():
                    aie.use_lock(write_out_locks[i], AcquireGreaterEqual, value=1)
                    aie.dma_bd(
                        buffer_0_1_c,
                        len=K,
                        offset=i * N_WRITE_OUTS * K,
                        iteration=(N_WRITE_OUTS, K),
                    )
                    aie.use_lock(read_in_locks[i], Release, value=1)

            aie.end()

    assert ctx.module.operation.verify()
    # print(ctx.module)

    compile_without_vectorization(ctx.module, workdir)

    buffer_args = {
        f"receive_a_col_{i}": (iters * N_WRITE_OUTS * K,) for i in range(N_CONSUMERS)
    }
    kernel_json = emit_design_kernel_json(buffer_args=list(buffer_args.keys()))
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(list(buffer_args.values()), np_dtype)
        for col, shim_channel_idx in col_shim_channel_index.items():
            bd_id = 0
            ipu_insts.extend(
                aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                    aiex.ipu.writebd_shimtile(
                        col,
                        bd_id=bd_id,
                        length=K,
                        iteration_size=(iters * N_WRITE_OUTS) - 1,
                        iteration_stride=K,
                    ),
                    tensor_addr=xclbin._get_buffer_host_address(buffer_idx=col),
                )
            )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(
                    channel_dir=S2MM,
                    channel_index=shim_channel_idx,
                    column=col,
                    bd_id=bd_id,
                    repeats=(iters * N_WRITE_OUTS) - 1,
                )
            )

        for col, shim_channel_idx in col_shim_channel_index.items():
            ipu_insts.extend(aiex.ipu.sync(channel=shim_channel_idx, column=col))

        xclbin.load_ipu_instructions(ipu_insts)

        wrapped = list(map(np.asarray, views))
        for w in wrapped:
            np.copyto(w, -1 * np.ones((N_WRITE_OUTS * iters * K), dtype=np.int32))

        xclbin.sync_buffers_to_device()
        start = time.monotonic_ns()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        end = time.monotonic_ns()
        print(f"time={(end - start) / 1e3}us")
        xclbin.sync_buffers_from_device()

        # correct = np.arange(1, iters + 1).repeat(K).reshape(iters, K).flatten()
        #
        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for w in wrapped:
                print(w.reshape(iters, N_WRITE_OUTS, K))


def test_single_prod_mult_cons_with_multiple_sync_bds_different_periods(
    ctx: MLIRContext, workdir: Path
):

    K = 4
    iters = 8
    ipu_insts = aiex.ipu.get_prolog()

    col_shim_channel_index = {}

    N_CONSUMERS = 4
    N_WRITE_OUTS = 4

    dtype = T.i32()
    np_dtype = np.int32
    byte_width_dtype = dtype.width // 8

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        tile_1_0 = aie.tile(1, 0)
        tile_2_0 = aie.tile(2, 0)
        tile_3_0 = aie.tile(3, 0)

        buffer_weight = aie.buffer(tile_0_2, (N_CONSUMERS * N_WRITE_OUTS * K,), T.i32())
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        core_to_mem_fl = aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)

        @aie.core(tile_0_2)
        def core():
            for i in range_(iters):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    for j in range(N_CONSUMERS):
                        ii = i + 1
                        jj = j + 1
                        linalg.fill(
                            ii + jj + ii * jj,
                            buffer_weight[
                                j * N_WRITE_OUTS * K : (j + 1) * N_WRITE_OUTS * K
                            ],
                        )
                yield_([])

        @aie.mem(tile_0_2)
        def mem_0_2():
            @aie.dma(MM2S, core_to_mem_fl.source_channel, repeat_count=iters - 1)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

        # try to use different channels as much as possible to prevent false positives
        mem_to_shim_tile_0_0_fl = aie.flow(tile_0_1, DMA, 1, tile_0_0, DMA, 0)
        col_shim_channel_index[0] = int(mem_to_shim_tile_0_0_fl.dest_channel)

        mem_to_shim_tile_1_0_fl = aie.flow(tile_0_1, DMA, 2, tile_1_0, DMA, 1)
        col_shim_channel_index[1] = int(mem_to_shim_tile_1_0_fl.dest_channel)

        mem_to_shim_tile_2_0_fl = aie.flow(tile_0_1, DMA, 3, tile_2_0, DMA, 0)
        col_shim_channel_index[2] = int(mem_to_shim_tile_2_0_fl.dest_channel)

        mem_to_shim_tile_3_0_fl = aie.flow(tile_0_1, DMA, 4, tile_3_0, DMA, 0)
        col_shim_channel_index[3] = int(mem_to_shim_tile_3_0_fl.dest_channel)

        consumer_flows = [
            mem_to_shim_tile_0_0_fl,
            mem_to_shim_tile_1_0_fl,
            mem_to_shim_tile_2_0_fl,
            mem_to_shim_tile_3_0_fl,
        ]

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            buffer_0_1_c = aie.buffer(
                tile_0_1, (N_CONSUMERS * N_WRITE_OUTS * K,), dtype
            )

            read_in_locks = [
                aie.lock(
                    tile_0_1,
                    init=N_WRITE_OUTS if i % 2 else 2 * N_WRITE_OUTS,
                    sym_name=f"read_in_{i}",
                )
                for i in range(N_CONSUMERS)
            ]
            write_out_locks = [
                aie.lock(tile_0_1, init=0, sym_name=f"write_out_{i}")
                for i in range(N_CONSUMERS)
            ]

            # read in
            @aie.dma(
                S2MM,
                core_to_mem_fl.dest_channel,
                repeat_count=iters - 1,
                num_bds=N_CONSUMERS,
            )
            def dma5():
                aie.use_lock(
                    read_in_locks[0],
                    AcquireGreaterEqual,
                    value=int(read_in_locks[0].owner.opview.init),
                )
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(
                    write_out_locks[0],
                    Release,
                    value=int(read_in_locks[0].owner.opview.init),
                )

            for i in range(1, N_CONSUMERS):
                write_outs = int(read_in_locks[i].owner.opview.init)

                @aie.another_bd(dma5)
                def _():
                    aie.use_lock(
                        read_in_locks[i], AcquireGreaterEqual, value=write_outs
                    )
                    aie.dma_bd(buffer_0_1_c, len=0)
                    aie.use_lock(write_out_locks[i], Release, value=write_outs)

            # write out
            for i, fl in enumerate(consumer_flows):
                write_outs = int(read_in_locks[i].owner.opview.init)
                k = K if i % 2 else K // 2

                @aie.dma(MM2S, fl.source_channel, repeat_count=(iters * write_outs) - 1)
                def dma6():
                    aie.use_lock(write_out_locks[i], AcquireGreaterEqual, value=1)
                    aie.dma_bd(
                        buffer_0_1_c,
                        len=k,
                        offset=i * write_outs * k,
                        iteration=(write_outs, k),
                    )
                    aie.use_lock(read_in_locks[i], Release, value=1)

            aie.end()

    assert ctx.module.operation.verify()
    print(ctx.module)

    compile_without_vectorization(ctx.module, workdir)

    buffer_args = {
        f"receive_a_col_{i}": (iters * N_WRITE_OUTS * K,) for i in range(N_CONSUMERS)
    }
    kernel_json = emit_design_kernel_json(buffer_args=list(buffer_args.keys()))
    xclbin_path = make_xclbin(ctx.module, workdir, kernel_json=kernel_json)

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers(list(buffer_args.values()), np_dtype)
        for i, (col, shim_channel_idx) in enumerate(col_shim_channel_index.items()):
            write_outs = N_WRITE_OUTS if i % 2 else 2 * N_WRITE_OUTS
            k = K if i % 2 else K // 2
            bd_id = 0
            ipu_insts.extend(
                aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                    aiex.ipu.writebd_shimtile(
                        col,
                        bd_id=bd_id,
                        length=k,
                        iteration_size=(iters * write_outs) - 1,
                        iteration_stride=k,
                    ),
                    tensor_addr=xclbin._get_buffer_host_address(buffer_idx=col),
                )
            )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(
                    channel_dir=S2MM,
                    channel_index=shim_channel_idx,
                    column=col,
                    bd_id=bd_id,
                    repeats=(iters * write_outs) - 1,
                )
            )

        for col, shim_channel_idx in col_shim_channel_index.items():
            ipu_insts.extend(aiex.ipu.sync(channel=shim_channel_idx, column=col))

        xclbin.load_ipu_instructions(ipu_insts)

        wrapped = list(map(np.asarray, views))
        for w in wrapped:
            np.copyto(w, -1 * np.ones((N_WRITE_OUTS * iters * K), dtype=np.int32))

        xclbin.sync_buffers_to_device()
        start = time.monotonic_ns()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        end = time.monotonic_ns()
        print(f"time={(end - start) / 1e3}us")
        xclbin.sync_buffers_from_device()

        # correct = np.arange(1, iters + 1).repeat(K).reshape(iters, K).flatten()
        #
        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            for w in wrapped:
                print(w.reshape(iters, N_WRITE_OUTS, K))
