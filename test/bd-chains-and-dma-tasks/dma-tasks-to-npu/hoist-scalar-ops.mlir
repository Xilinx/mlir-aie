//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A builder that derives BD operands from runtime scalars emits the casts, the
// transfer-length product and its npu.require guard where it is called, inside
// the BD block. The lowering moves them ahead of the task in program order,
// cloning the constants they read so the dma_bd keeps its own.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: %[[N64:.*]] = arith.extsi %{{.*}} : i32 to i64
// CHECK: %[[U64:.*]] = emitc.cast %{{.*}} : ui32 to i64
// CHECK: %[[PROD:.*]] = arith.muli %[[N64]], %{{.*}} : i64
// CHECK: %[[OK:.*]] = arith.cmpi ule, %[[PROD]], %{{.*}} : i64
// CHECK: aiex.npu.require(%[[OK]]) {message = "a runtime DMA transfer length does not fit in 32 bits"}
// CHECK: arith.trunci %[[PROD]] : i64 to i32
// CHECK: aiex.npu.require(%{{.*}}) {message = "a runtime DMA size or stride does not fit in 31 bits"}
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 1048575 : i32}
// CHECK: aiex.npu.blockwrite_values
// CHECK: aiex.npu.address_patch
// CHECK-NOT: aiex.dma_configure_task

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<8192xi32>, %n: i32, %s: ui32) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        %c32 = arith.constant 32 : i64
        %max = arith.constant 4294967295 : i64
        %n64 = arith.extsi %n : i32 to i64
        %s64 = emitc.cast %s : ui32 to i64
        %len64 = arith.muli %n64, %c32 : i64
        %fits = arith.cmpi ule, %len64, %max : i64
        aiex.npu.require(%fits) {message = "a runtime DMA transfer length does not fit in 32 bits"} : i1
        %len = arith.trunci %len64 : i64 to i32
        aie.dma_bd(%arg0 : memref<8192xi32> offset = 0 len = %len sizes = [1, 1, %n64, %c32] strides = [0, 0, %s64, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
