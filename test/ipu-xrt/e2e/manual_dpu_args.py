# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

# RUN: VITIS_DIR=$VITIS WORKDIR=$PWD XRT_DIR=%XRT_DIR %PYTHON %s

import sys

# this is to get the MemRefValue caster inside of aie-python-extras
# noinspection PyUnresolvedReferences
from aie.extras.dialects.ext import arith, func, linalg, memref
from filelock import FileLock
import numpy as np

from aie.compiler.aiecc.main import emit_design_kernel_json
from aie.dialects import aie, aiex
from aie.dialects.aie import (
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
)
import aie.extras.types as T
from aie.xrt import XCLBin
from util import (
    compile_without_vectorization,
    construct_and_print_module,
    make_xclbin,
)

DMA = WireBundle.DMA
S2MM = DMAChannelDir.S2MM
MM2S = DMAChannelDir.MM2S
Acquire = LockAction.Acquire
AcquireGreaterEqual = LockAction.AcquireGreaterEqual
Release = LockAction.Release


# CHECK-LABEL: manual_args
@construct_and_print_module
def manual_args(module):
    K = 32
    RANDOM_WEIGHT = np.random.randint(0, 10, (K,), dtype=np.int32)
    repeat_count = 10
    loop = False

    @aie.device(AIEDevice.ipu)
    def ipu():
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
            for j in range(repeat_count):
                with aiex.hold_lock(lock_read_weight, lock_send_weight):
                    linalg.fill(j, y)
                    linalg.copy(x, buffer_weight)
                    linalg.add(y, buffer_weight, buffer_weight)

        @aie.mem(tile_0_2)
        def mem_0_2():
            @aie.dma(MM2S, 0, loop=loop, repeat_count=repeat_count)
            def dma3():
                aie.use_lock(lock_send_weight, AcquireGreaterEqual)
                # TODO(max): would prefer to be able to stick get_global here...
                aie.dma_bd(buffer_weight)
                aie.use_lock(lock_read_weight, Release)

            aie.end()

        @aie.memtile_dma(tile_0_1)
        def memtile_dma_0_1():
            lock_0_1_read_in_c = aie.lock(tile_0_1, init=1)
            lock_0_1_write_out_c = aie.lock(tile_0_1, init=0)
            buffer_0_1_c = aie.buffer(tile_0_1, (K,), T.i32())

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

    assert module.operation.verify()

    compile_without_vectorization(module)
    buffer_args = [f"out_{i}" for i in range(repeat_count)]
    kernel_json = emit_design_kernel_json(buffer_args=buffer_args)
    xclbin_path = make_xclbin(module, kernel_json=kernel_json)

    with FileLock("/tmp/ipu.lock"):
        xclbin = XCLBin(xclbin_path, "MLIR_AIE")
        views = xclbin.mmap_buffers([(K,)] * repeat_count, np.int32)

        col = 0
        channel_index = 0
        ipu_insts = aiex.ipu.get_prolog()
        for bd_id in range(repeat_count):
            writebd_shimtile_insts = aiex.ipu.writebd_shimtile(bd_id, buffer_length=K)
            ipu_insts.extend(
                aiex.ipu._exec_write_bd_extend_shim_tile_opt(
                    writebd_shimtile_insts,
                    tensor_addr=xclbin._get_buffer_host_address(bd_id),
                )
            )
            ipu_insts.extend(
                aiex.ipu.shimtile_push_queue(S2MM, channel_index, col, bd_id)
            )
            ipu_insts.extend(aiex.ipu.sync(column=col))
        assert all(i < 2**32 for i in ipu_insts)

        xclbin.load_ipu_instructions(ipu_insts)

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
