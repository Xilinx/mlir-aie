# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: VITIS_DIR=$VITIS WORKDIR=$PWD XRT_DIR=%XRT_DIR %PYTHON %s

import sys

from aie.dialects import aie, aiex
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref
import aie.extras.types as T
from aie.ir import _i32ElementsAttr
from aie.xrt import XCLBin
from filelock import FileLock
import numpy as np

from util import (
    compile_without_vectorization,
    construct_and_print_module,
    make_xclbin,
    setup_xclbin_firmware,
)

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: repeat_count
@construct_and_print_module
def repeat_count(module):
    K = 32
    repeat_count = 4
    loop = False
    RANDOM_WEIGHT = np.random.randint(0, 10, (K,), dtype=np.int32)
    ipu_insts = aiex.ipu.get_prolog()

    @aie.device(AIEDevice.ipu)
    def ipu():
        tile_0_0 = aie.tile(0, 0)
        tile_0_1 = aie.tile(0, 1)
        tile_0_2 = aie.tile(0, 2)

        weight = memref.global_(initial_value=RANDOM_WEIGHT, constant=True)
        buffer_weight = aie.buffer(T.memref(K, T.i32()), tile_0_2)
        lock_read_weight = aie.lock(tile_0_2, init=1)
        lock_send_weight = aie.lock(tile_0_2, init=0)

        aie.flow(tile_0_2, DMA, 0, tile_0_1, DMA, 2)
        aie.flow(tile_0_1, DMA, 2, tile_0_0, DMA, 0)

        @aie.core(tile_0_2)
        def core():
            x = memref.get_global(weight.type_.value, weight.sym_name.value)
            for _ in range(repeat_count):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.copy(x, buffer_weight)

        @aie.mem(tile_0_2)
        def mem_0_2():
            @aie.dma(MM2S, 0, loop=loop, repeat_count=repeat_count)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            lock_0_1_read_in_c = aie.lock(tile_0_1, init=1)
            lock_0_1_write_out_c = aie.lock(tile_0_1, init=0)
            buffer_0_1_c = aie.buffer(T.memref(K, T.i32()), tile_0_1)

            @aie.dma(S2MM, 2, loop=loop, repeat_count=repeat_count)
            def dma5():
                aie.use_lock(lock_0_1_read_in_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_write_out_c, Release)

            @aie.dma(MM2S, 2, loop=loop, repeat_count=repeat_count)
            def dma6():
                aie.use_lock(lock_0_1_write_out_c, AcquireGreaterEqual)
                aie.dma_bd(buffer_0_1_c)
                aie.use_lock(lock_0_1_read_in_c, Release)

            aie.end()

        # in A
        channel_index = 0
        ddr_id = 0
        col = 0
        for i, bd_id in enumerate(range(repeat_count)):
            ipu_insts.extend(
                aiex.ipu.writebd_shimtile(
                    bd_id,
                    buffer_length=K,
                    buffer_offset=i * K,
                    ddr_id=ddr_id,
                )
            )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            ipu_insts.extend(
                aiex.ipu.sync(
                    channel=0,
                    column=0,
                    column_num=1,
                    direction=0,
                    row=0,
                    row_num=1,
                )
            )

    assert module.operation.verify()

    compile_without_vectorization(module)
    xclbin_path = make_xclbin(module)
    with FileLock("/tmp/ipu.lock"):
        setup_xclbin_firmware(xclbin_path)

        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        xclbin.load_ipu_instructions(ipu_insts)
        views = xclbin.mmap_buffers([(repeat_count * K,)], np.int32)

        wrap_C = np.asarray(views[0])

        C = np.zeros((repeat_count * K,), dtype=np.int32)
        np.copyto(wrap_C, C, casting="no")

        xclbin.sync_buffers_to_device()
        xclbin.run()
        print("Running kernel")
        xclbin.wait(30)
        xclbin.sync_buffers_from_device()

        if not np.array_equal(np.concatenate([RANDOM_WEIGHT] * repeat_count), wrap_C):
            with np.printoptions(threshold=sys.maxsize, linewidth=sys.maxsize):
                print(RANDOM_WEIGHT)
                print(wrap_C)
                assert False
