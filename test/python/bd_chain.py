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
        # CHECK: %[[tile0:.+]] = aie.tile
        shim_tile = tile(0, 0)
        # CHECK: %[[tile1:.+]] = aie.tile
        mem_tile = tile(0, 1)

        of0 = object_fifo("of0", shim_tile, mem_tile, 2, T.memref(256, T.i32()))

        @bd_chain(T.memref(16, T.i16()), T.memref(32, T.i16()))
        def my_chain(bd, a, b):
            with bd[0]:
                dma_bd(a)
                next_bd(bd[1])
            with bd[1]:
                dma_bd(b)
                EndOp()

        @runtime_sequence(T.memref(16, T.i16()), T.memref(32, T.i16()))
        def seq(a, b):
            # CHECK: %[[tsk0:.+]] = aiex.dma_start_bd_chain @my_chain({{.+}}) on (%[[tile0]], MM2S, 1)
            tsk0 = dma_start_bd_chain(
                my_chain, [a, b], shim_tile, DMAChannelDir.MM2S, 1
            )
            # CHECK: %[[tsk1:.+]] = aiex.dma_start_bd_chain_for @my_chain({{.+}}) for{{ +}}@of0
            tsk1 = dma_start_bd_chain_for(my_chain, [a, b], of0)

    print(ctx.module)
