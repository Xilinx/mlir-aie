# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu --aie-dma-to-npu | FileCheck %s --check-prefix=LOWERED
import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu1_1col)
    def device_body():
        # CHECK: %[[shim:.+]] = aie.tile(0, 0)
        shim = tile(0, 0)

        # CHECK: aie.runtime_sequence @seq(%[[BUF:.+]]: memref<128xi32>, %[[M:.+]]: i64, %[[S:.+]]: i64)
        @runtime_sequence(T.memref(128, T.i32()), T.i64(), T.i64())
        def seq(a, M, S):
            # All-static fast path - the existing static printer form.
            # CHECK: aie.dma_bd(%[[BUF]] : memref<128xi32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4, stride = 32>, <size = 32, stride = 1>]) {bd_id = 0 : i32}
            t1 = dma_configure_task(shim, DMAChannelDir.MM2S, 0)
            with bds(t1) as bd:
                with bd[0]:
                    dma_bd(
                        a,
                        offset=0,
                        len=128,
                        sizes=[1, 1, 4, 32],
                        strides=[0, 0, 32, 1],
                        bd_id=0,
                    )
                    EndOp()
            dma_start_task(t1)

            # SSA offset only - dyn_offset routes through, dimensions stay static.
            # CHECK: aie.dma_bd(%[[BUF]] : memref<128xi32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4, stride = 32>, <size = 32, stride = 1>]) dyn_offset(%[[M]] : i64) {bd_id = 1 : i32}
            t2 = dma_configure_task(shim, DMAChannelDir.MM2S, 0)
            with bds(t2) as bd:
                with bd[0]:
                    dma_bd(
                        a,
                        offset=M,
                        len=128,
                        sizes=[1, 1, 4, 32],
                        strides=[0, 0, 32, 1],
                        bd_id=1,
                    )
                    EndOp()
            dma_start_task(t2)

            # Fully dynamic sizes/strides/len - static dims become arith.constant
            # hoisted just before the configure_task; dimensions attribute is dropped.
            # CHECK: arith.constant
            # CHECK: aie.dma_bd(%[[BUF]] : memref<128xi32>) dyn_len(%[[M]] : i64) dyn_sizes
            # CHECK-SAME: dyn_strides
            t3 = dma_configure_task(shim, DMAChannelDir.MM2S, 0)
            with bds(t3) as bd:
                with bd[0]:
                    dma_bd(
                        a,
                        offset=0,
                        len=M,
                        sizes=[1, 1, M, S],
                        strides=[0, 0, S, 1],
                        bd_id=2,
                    )
                    EndOp()
            dma_start_task(t3)


# LOWERED checks: after lowering, dma_configure_task_for → blockwrite + write32
# LOWERED: aiex.npu.blockwrite
# LOWERED: aiex.npu.write32
# LOWERED-NOT: aiex.dma_configure_task

print(ctx.module)
