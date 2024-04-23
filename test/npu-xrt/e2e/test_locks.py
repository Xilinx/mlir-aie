# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from pathlib import Path
import random
import sys

from aie.compiler.util import (
    compile_without_vectorization,
    make_xclbin,
)
from aie.dialects import aie, aiex
from aie.dialects.aie import AIEDevice, DMAChannelDir

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref

# noinspection PyUnresolvedReferences
from aie.extras.testing import MLIRContext, filecheck, mlir_ctx as ctx
import aie.extras.types as T
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np
import pytest

# needed since the fix isn't defined here nor conftest.py
pytest.mark.usefixtures("ctx")

pytest.mark.usefixtures("run_around_tests")

S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S


def test_one_global(ctx: MLIRContext, workdir: Path):
    K = 32
    RANDOM_NUMBER = random.randint(0, 100)
    iv = np.random.randint(0, 10, (K,), dtype=np.int32)
    column = 2

    npu_insts = aiex.npu.get_prolog()

    @aie.device(AIEDevice.npu)
    def npu():
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
        npu_insts.extend(
            aiex.npu.writebd_shimtile(
                column=column,
                bd_id=bd_id,
                buffer_length=K,
                buffer_offset=0,
                ddr_id=ddr_id,
            )
        )
        npu_insts.extend(
            aiex.npu.shimtile_push_queue(
                channel_dir=S2MM,
                channel_index=flow_to_shim.dest_channel,
                column=column,
                bd_id=bd_id,
            )
        )
        npu_insts.extend(
            aiex.npu.sync(
                channel=flow_to_shim.dest_channel,
                column=column,
                direction=0,
                row=0,
            )
        )

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_npu_instructions(npu_insts)
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

    npu_insts = aiex.npu.get_prolog()

    @aie.device(AIEDevice.npu)
    def npu():
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
        npu_insts.extend(
            aiex.npu.writebd_shimtile(
                column=shim_tile_column,
                bd_id=bd_id,
                buffer_length=K,
                buffer_offset=0,
                ddr_id=ddr_id,
            )
        )
        npu_insts.extend(
            aiex.npu.shimtile_push_queue(
                channel_dir=S2MM,
                channel_index=flow_to_shim.dest_channel,
                column=shim_tile_column,
                bd_id=bd_id,
            )
        )
        npu_insts.extend(
            aiex.npu.sync(
                channel=flow_to_shim.dest_channel,
                column=shim_tile_column,
                direction=0,
                row=0,
            )
        )

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_npu_instructions(npu_insts)
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

    npu_insts = aiex.npu.get_prolog()

    @aie.device(AIEDevice.npu)
    def npu():
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
        npu_insts.extend(
            aiex.npu.writebd_shimtile(
                column=shim_tile_column,
                bd_id=bd_id,
                buffer_length=K,
                buffer_offset=0,
                ddr_id=ddr_id,
            )
        )
        npu_insts.extend(
            aiex.npu.shimtile_push_queue(
                channel_dir=S2MM,
                channel_index=flow_to_shim.dest_channel,
                column=shim_tile_column,
                bd_id=bd_id,
            )
        )
        npu_insts.extend(
            aiex.npu.sync(
                channel=flow_to_shim.dest_channel,
                column=shim_tile_column,
                direction=0,
                row=0,
            )
        )

    compile_without_vectorization(ctx.module, workdir)
    xclbin_path = make_xclbin(ctx.module, workdir)
    with FileLock("/tmp/npu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_npu_instructions(npu_insts)
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
