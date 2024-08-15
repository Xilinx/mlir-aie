# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.ir as ir


with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu1_1col)
    def device_body():
        shim_tile = tile(0, 0)
        # CHECK: %[[tile:.+]] = aie.tile
        mem_tile = tile(0, 1)

        # CHECK: %[[buf0:.+]] = aie.buffer
        buf0 = buffer(mem_tile, (16,), T.i16())
        # CHECK: %[[buf1:.+]] = aie.buffer
        buf1 = buffer(mem_tile, (32,), T.i16())

        of0 = object_fifo("of0", shim_tile, mem_tile, 2, T.memref(256, T.i32()))

        @runtime_sequence(T.memref(256, T.i32()))
        def seq(a):
            # CHECK: %dma_configure_task_0_1 = aiex.dma_configure_task(%[[tile]], MM2S, 1) {
            tsk0 = dma_configure_task(mem_tile, DMAChannelDir.MM2S, 1)
            with bds(tsk0) as bd:
                with bd[0]:
                    # CHECK:   aie.next_bd ^bb1
                    next_bd(bd[1])
                    pass
                # CHECK: ^bb1:
                with bd[1]:
                    # CHECK:   aie.dma_bd(%[[buf0]] : memref<16xi16>)
                    dma_bd(buf0)
                    # CHECK:   aie.next_bd ^bb2
                    next_bd(bd[2])
                # CHECK: ^bb2:
                with bd[2]:
                    # CHECK:   aie.dma_bd(%[[buf1]] : memref<32xi16>)
                    dma_bd(buf1)
                    # CHECK:   aie.next_bd ^bb1
                    next_bd(bd[1])

            tsk1 = dma_configure_task_for(of0)
            with bds(tsk1) as bd:
                with bd[0]:
                    dma_bd(a)
                    EndOp()

    print(ctx.module)
