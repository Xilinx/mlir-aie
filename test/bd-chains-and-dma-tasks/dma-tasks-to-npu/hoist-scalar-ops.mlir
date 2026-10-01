//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

// A builder that derives BD operands from runtime scalars emits the casts, the
// transfer-length product and its cf.assert guard where it is called, inside
// the BD block. The lowering moves them ahead of the task in program order
// (loop-invariant code motion out of the task body), so the BD words and the
// lowering's own guards can use them.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: %[[N64:.*]] = arith.extsi %{{.*}} : i32 to i64
// CHECK: %[[S64:.*]] = arith.extui %{{.*}} : i32 to i64
// CHECK: %[[PROD:.*]] = arith.muli %[[N64]], %{{.*}} : i64
// CHECK: %[[OK:.*]] = arith.cmpi ule, %[[PROD]], %{{.*}} : i64
// CHECK: cf.assert %[[OK]], "a runtime DMA transfer length does not fit in 32 bits"
// CHECK: arith.trunci %[[PROD]] : i64 to i32
// CHECK: cf.assert %{{.*}}, "a runtime DMA d1 size must be in [1:1023]"
// CHECK: cf.assert %{{.*}}, "a runtime DMA d1 stride must be in [1:1048576] when its size > 1"
// CHECK: cf.assert %{{.*}}, "a runtime DMA length must equal the d0*d1*d2 extent of its dimensions"
// CHECK: cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 8192-element host buffer"
// CHECK: aiex.npu.blockwrite_values
// CHECK: aiex.npu.address_patch
// CHECK-NOT: aiex.dma_configure_task

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<8192xi32>, %n: i32, %s: i32) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        %c32 = arith.constant 32 : i64
        %max = arith.constant 4294967295 : i64
        %n64 = arith.extsi %n : i32 to i64
        %s64 = arith.extui %s : i32 to i64
        %len64 = arith.muli %n64, %c32 : i64
        %fits = arith.cmpi ule, %len64, %max : i64
        cf.assert %fits, "a runtime DMA transfer length does not fit in 32 bits"
        %len = arith.trunci %len64 : i64 to i32
        aie.dma_bd(%arg0 : memref<8192xi32> offset = 0 len = %len sizes = [1, 1, %n64, %c32] strides = [0, 0, %s64, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}

// -----

// A chained task: the second BD block computes from a value the first block
// defined. Both blocks' scalar ops leave the task in their original order,
// so the use still follows its definition.

// CHECK-LABEL: aie.runtime_sequence @chained
// CHECK: %[[N64:.*]] = arith.extui %arg1 : i32 to i64
// CHECK: %[[TWICE:.*]] = arith.muli %[[N64]], %{{.*}} : i64
// CHECK: cf.assert %{{.*}}, "n must be at most 512"
// CHECK-COUNT-2: aiex.npu.blockwrite_values
// CHECK-NOT: aiex.dma_configure_task

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence @chained(%arg0: memref<8192xi32>, %n: i32) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        %c32 = arith.constant 32 : i64
        %n64 = arith.extui %n : i32 to i64
        aie.dma_bd(%arg0 : memref<8192xi32> offset = 0 len = 1024 sizes = [1, 1, %n64, %c32] strides = [0, 0, 32, 1]) {bd_id = 0 : i32}
        aie.next_bd ^bd1
      ^bd1:
        %c2 = arith.constant 2 : i64
        %c512 = arith.constant 512 : i64
        %c16 = arith.constant 16 : i64
        %twice = arith.muli %n64, %c2 : i64
        %ok = arith.cmpi ule, %n64, %c512 : i64
        cf.assert %ok, "n must be at most 512"
        aie.dma_bd(%arg0 : memref<8192xi32> offset = 0 len = 1024 sizes = [1, 1, %twice, %c16] strides = [0, 0, 16, 1]) {bd_id = 1 : i32}
        aie.end
      }
    }
  }
}
