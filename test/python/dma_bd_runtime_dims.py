# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids --aie-dma-tasks-to-npu | FileCheck %s --check-prefix=NPU

# dma_bd() and shim_dma_bd() called directly inside a BD block with i32
# runtime dimensions: the i64 widening and the default transfer length are
# emitted ahead of the task, since the BD block may hold only dma_bd/aie.end.
# An i64 runtime offset/length is range-guarded and truncated to the op's i32
# operand, also ahead of the task. Unsigned runtime values are zero-extended.

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.taplib import TensorAccessPattern

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu1_1col)
    def device_body():
        shim = tile(0, 0)
        shim_dma_allocation("of_in", shim, DMAChannelDir.MM2S, 0)

        @runtime_sequence(
            T.memref(4096, T.i32()),
            T.i32(),
            T.i64(),
            IntegerType.get_unsigned(8),
            IntegerType.get_unsigned(32),
        )
        def seq(a, n, m, u8, u32):
            t = dma_configure_task_for("of_in")
            with bds(t) as bd:
                with bd[0]:
                    dma_bd(
                        a,
                        sizes=[1, 1, n, 16],
                        strides=[0, 0, 16, 1],
                        offset=0,
                        transfer_len=16,
                    )
                    EndOp()
            dma_start_task(t)
            dma_free_task(t)

            tap = TensorAccessPattern((4096,), 0, [1, 1, n, 16], [0, 0, 16, 1])
            t2 = dma_configure_task_for("of_in")
            with bds(t2) as bd:
                with bd[0]:
                    shim_dma_bd(a, tap=tap)
                    EndOp()
            dma_start_task(t2)
            dma_free_task(t2)

            t3 = dma_configure_task_for("of_in")
            with bds(t3) as bd:
                with bd[0]:
                    dma_bd(a, offset=m, transfer_len=m)
                    EndOp()
            dma_start_task(t3)
            dma_free_task(t3)

            t4 = dma_configure_task_for("of_in")
            with bds(t4) as bd:
                with bd[0]:
                    dma_bd(
                        a,
                        sizes=[1, 1, u8, 16],
                        strides=[0, 0, 16, 1],
                        offset=u32,
                        transfer_len=u8,
                    )
                    EndOp()
            dma_start_task(t4)
            dma_free_task(t4)

    print(ctx.module)

# CHECK-LABEL: aie.runtime_sequence
# CHECK: %[[W0:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK-NEXT: aiex.dma_configure_task_for @of_in {
# CHECK-NEXT: aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 16 sizes = [1, 1, %[[W0]], 16] strides = [0, 0, 16, 1])
# CHECK-NEXT: aie.end

# CHECK: arith.muli
# CHECK: %[[W1:.*]] = arith.extsi %arg1 : i32 to i64
# CHECK-NEXT: aiex.dma_configure_task_for @of_in {
# CHECK-NEXT: aie.dma_bd(%arg0 : memref<4096xi32> offset = {{.*}} len = %{{[0-9]+}} sizes = [1, 1, %[[W1]], 16] strides = [0, 0, 16, 1])
# CHECK-NEXT: aie.end

# CHECK: %[[OK0:.*]] = arith.cmpi ule, %arg2, %{{.*}} : i64
# CHECK-NEXT: aiex.npu.require(%[[OK0]]) {message = "a runtime DMA offset does not fit its 32-bit field"}
# CHECK-NEXT: %[[OFF:.*]] = arith.trunci %arg2 : i64 to i32
# CHECK: %[[OK1:.*]] = arith.cmpi ule, %arg2, %{{.*}} : i64
# CHECK-NEXT: aiex.npu.require(%[[OK1]]) {message = "a runtime DMA transfer length does not fit its 32-bit field"}
# CHECK-NEXT: %[[LEN:.*]] = arith.trunci %arg2 : i64 to i32
# CHECK-NEXT: aiex.dma_configure_task_for @of_in {
# CHECK-NEXT: aie.dma_bd(%arg0 : memref<4096xi32> offset = %[[OFF]] len = %[[LEN]])
# CHECK-NEXT: aie.end

# CHECK: %[[U0:.*]] = emitc.cast %arg3 : ui8 to i64
# CHECK-NEXT: %[[UOFF:.*]] = emitc.cast %arg4 : ui32 to i32
# CHECK-NEXT: %[[ULEN:.*]] = emitc.cast %arg3 : ui8 to i32
# CHECK-NEXT: aiex.dma_configure_task_for @of_in {
# CHECK-NEXT: aie.dma_bd(%arg0 : memref<4096xi32> offset = %[[UOFF]] len = %[[ULEN]] sizes = [1, 1, %[[U0]], 16] strides = [0, 0, 16, 1])
# CHECK-NEXT: aie.end

# NPU-LABEL: aie.runtime_sequence
# NPU-NOT: aiex.dma_configure_task_for
# NPU: aiex.npu.blockwrite_values
# NPU: aiex.npu.blockwrite_values
# NPU: aiex.npu.require
# NPU: aiex.npu.require
# NPU: aiex.npu.blockwrite_values
# NPU: emitc.cast %arg3 : ui8 to i64
# NPU: aiex.npu.blockwrite_values
