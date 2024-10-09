# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.helpers.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *


with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu1_1col)
    def device_body():
        # CHECK: %[[tile_0_2:.+]] = aie.tile
        tile_0_2 = tile(0, 2)
        # CHECK: %[[buf0:.+]] = aie.buffer
        buf0 = buffer(tile_0_2, (16,), T.i32())
        # CHECK: %[[buf1:.+]] = aie.buffer
        buf1 = buffer(tile_0_2, (16,), T.i32())
        # CHECK: %[[lock0:.+]] = aie.lock
        lock0 = lock(tile_0_2)
        # CHECK: %[[lock1:.+]] = aie.lock
        lock1 = lock(tile_0_2)

        # CHECK: %{{.+}} = aie.mem(%[[tile_0_2]]) {
        # CHECK:    %0 = aie.dma_start(S2MM, 1, ^bb1, ^bb2)
        # CHECK:  ^bb1:  // pred: ^bb0
        # CHECK:    aie.use_lock(%[[lock0]], AcquireGreaterEqual, 1)
        # CHECK:    aie.dma_bd(%[[buf0]] : memref<16xi32>)
        # CHECK:    aie.use_lock(%[[lock1]], Release, 1)
        # CHECK:    aie.end
        # CHECK:  ^bb2:  // pred: ^bb0
        # CHECK:    %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
        # CHECK:  ^bb3:  // pred: ^bb2
        # CHECK:    aie.use_lock(%[[lock1]], AcquireGreaterEqual, 1)
        # CHECK:    aie.dma_bd(%[[buf1]] : memref<16xi32>)
        # CHECK:    aie.use_lock(%[[lock0]], Release, 1)
        # CHECK:    aie.end
        # CHECK:  ^bb4:  // pred: ^bb2
        # CHECK:    aie.end
        @mem(tile_0_2)
        def m(block):
            s0 = dma_start(DMAChannelDir.S2MM, 1, dest=block[1], chain=block[2])
            with block[1]:
                use_lock(lock0, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf0)
                use_lock(lock1, LockAction.Release, value=1)
                EndOp()
            with block[2]:
                s1 = dma_start(DMAChannelDir.MM2S, 0, dest=block[3], chain=block[4])
            with block[3]:
                use_lock(lock1, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf1)
                use_lock(lock0, LockAction.Release, value=1)
                EndOp()
            with block[4]:
                EndOp()

    print(ctx.module)
