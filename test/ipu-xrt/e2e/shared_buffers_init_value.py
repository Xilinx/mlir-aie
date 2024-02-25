# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
# RUN: VITIS_DIR=$VITIS WORKDIR=$PWD XRT_DIR=%XRT_DIR %PYTHON %s
import random
import sys

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref
from filelock import FileLock
import numpy as np

from aie.dialects import aie, aiex
from aie.dialects.aie import AIEDevice, DMAChannelDir
import aie.extras.types as T
from aie.xrt import XCLBin
from util import (
    compile_without_vectorization,
    construct_and_print_module,
    make_xclbin,
)

S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S


# CHECK-LABEL: foursome
@construct_and_print_module
def foursome(module):
    K = 32
    iv1 = np.random.randint(0, 10, (K,), dtype=np.int32)
    iv11 = np.random.randint(0, 10, (K,), dtype=np.int32)
    RANDOM_NUMBER = random.randint(0, 100)
    iv2 = np.random.randint(0, 10, (K,), dtype=np.int32)
    iv3 = np.random.randint(0, 10, (K,), dtype=np.int32)
    iv4 = np.random.randint(0, 10, (K,), dtype=np.int32)

    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        _dummy_tile = aie.tile(0, 2)

        # west
        tile_1_3 = aie.tile(1, 3)
        # north
        tile_2_4 = aie.tile(2, 4)
        # south
        tile_2_2 = aie.tile(2, 2)
        # self
        tile_2_3 = aie.tile(2, 3)

        buffer_weight_1_3 = aie.buffer(tile_1_3, (K,), T.i32(), initial_value=iv1)
        buffer_weight_1_3_random_number = aie.buffer(tile_1_3, (K,), T.i32())
        buffer_weight_1_3_another_init = aie.buffer(
            tile_1_3, (K,), T.i32(), initial_value=iv4
        )
        lock_weight_1_3 = aie.lock(tile_1_3, init=0)

        @aie.core(tile_1_3)
        def core():
            with aiex.hold_lock(lock_weight_1_3, lock_weight_1_3, acq_val=0):
                linalg.fill(RANDOM_NUMBER, buffer_weight_1_3_random_number)
                linalg.add(
                    buffer_weight_1_3_random_number,
                    buffer_weight_1_3,
                    buffer_weight_1_3,
                )
                linalg.add(
                    buffer_weight_1_3_another_init, buffer_weight_1_3, buffer_weight_1_3
                )

        buffer_weight_2_4 = aie.buffer(tile_2_4, (K,), T.i32(), initial_value=iv2)
        lock_weight_2_4 = aie.lock(tile_2_4, init=0)

        @aie.core(tile_2_4)
        def core():
            with aiex.hold_lock(lock_weight_2_4, lock_weight_2_4, acq_val=0):
                linalg.add(buffer_weight_2_4, buffer_weight_2_4, buffer_weight_2_4)

        buffer_weight_2_2 = aie.buffer(tile_2_2, (K,), T.i32(), initial_value=iv3)
        lock_weight_2_2 = aie.lock(tile_2_2, init=0)

        @aie.core(tile_2_2)
        def core():
            with aiex.hold_lock(lock_weight_2_2, lock_weight_2_2, acq_val=0):
                linalg.add(buffer_weight_2_2, buffer_weight_2_2, buffer_weight_2_2)

        lock_use_weight_2_3 = aie.lock(tile_2_3, init=0)

        global_weight_1_3 = aie.external_buffer((K,), T.i32(), ref=buffer_weight_1_3)
        global_weight_2_4 = aie.external_buffer((K,), T.i32(), ref=buffer_weight_2_4)
        global_weight_2_2 = aie.external_buffer((K,), T.i32(), ref=buffer_weight_2_2)
        buffer_weight_2_3 = aie.buffer(tile_2_3, (K,), T.i32(), initial_value=iv4)

        @aie.core(tile_2_3)
        def core():
            with (
                aiex.hold_lock(lock_weight_1_3, lock_weight_1_3),
                aiex.hold_lock(lock_weight_2_2, lock_weight_2_2),
                aiex.hold_lock(lock_weight_2_4, lock_weight_2_4),
                aiex.hold_lock(lock_use_weight_2_3, lock_use_weight_2_3, acq_val=0),
            ):
                linalg.add(global_weight_1_3, buffer_weight_2_3, buffer_weight_2_3)
                linalg.add(global_weight_2_4, buffer_weight_2_3, buffer_weight_2_3)
                linalg.add(global_weight_2_2, buffer_weight_2_3, buffer_weight_2_3)

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
                bd_id=bd_id,
                column=shim_tile_column,
                buffer_length=K,
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

    compile_without_vectorization(module)
    xclbin_path = make_xclbin(module)
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

    if not np.array_equal(2 * iv1 + 2 * iv2 + 2 * iv3 + iv4, wrap_C):
        with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
            print(f"{iv1=}")
            print(f"{iv2=}")
            print(f"{iv3=}")
            print(f"{iv4=}")
            print(f"c={wrap_C}")
