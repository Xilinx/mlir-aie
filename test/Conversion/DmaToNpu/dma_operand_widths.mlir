//===- dma_operand_widths.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Runtime DMA operands take any signless integer or index. An i64 list entry
// (i32 for a dma_bd offset/len) keeps the untyped spelling; any other width
// carries its type. The lowering zero-extends each one to i64 before it is
// guarded, and guards a wider one to fit before truncating it.

// RUN: aie-opt %s | aie-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: aie-opt --aie-substitute-shim-dma-allocations --aie-dma-tasks-to-npu --aie-dma-to-npu --split-input-file --verify-diagnostics %s | FileCheck %s

// ROUNDTRIP-LABEL: @widths
// ROUNDTRIP: aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, %arg2 : index][1, 1, %arg1 : i32, %arg3][0, 0, 64, 1])
// ROUNDTRIP-LABEL: @task
// ROUNDTRIP: aiex.dma_configure_task_for @a repeat %arg2 : index
// ROUNDTRIP: aie.dma_bd(%arg0 : memref<4096xi32> offset = %arg2 : index len = %arg3 sizes = [%arg1 : i16, 64] strides = [64, 1])

// CHECK-LABEL: @widths
// CHECK: arith.extui %arg1 : i32 to i64
// CHECK: cf.assert %{{.*}}, "a runtime DMA d1 size must be in [1:1023]"
// CHECK: arith.index_castui %arg2 : index to i64
// CHECK: cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 4096-element host buffer"
// CHECK: aiex.npu.address_patch(%{{.*}} : i64)
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @widths(%in: memref<4096xi32>, %n: i32, %off: index, %one: i64) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, %off : index][1, 1, %n : i32, %one][0, 0, 64, 1]) {id = 0 : i64, metadata = @a} : memref<4096xi32>
    }
    aie.runtime_sequence @task(%in: memref<4096xi32>, %n: i16, %r: index, %len: i32) {
      %task = aiex.dma_configure_task_for @a repeat %r : index {
        aie.dma_bd(%in : memref<4096xi32> offset = %r : index len = %len sizes = [%n : i16, 64] strides = [64, 1]) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%task)
      aiex.dma_await_task(%task)
    }
  }
}

// CHECK-LABEL: @task
// CHECK: arith.extui %arg1 : i16 to i64
// CHECK: arith.extui %arg3 : i32 to i64
// CHECK: cf.assert %{{.*}}, "a runtime DMA access runs past the end of its 4096-element host buffer"
// CHECK: aiex.npu.address_patch
// CHECK: %[[R:.*]] = arith.index_castui %arg2 : index to i64
// CHECK: %[[OK:.*]] = arith.cmpi ule, %[[R]], %c255_i64 : i64
// CHECK: cf.assert %[[OK]], "a runtime DMA repeat count exceeds the task queue's [0:255] range (at most 256 executions)"
// CHECK: arith.trunci %[[R]] : i64 to i32
// CHECK: aiex.npu.write32

// -----

// CHECK-LABEL: @wide
// CHECK: %[[FITS:.*]] = arith.cmpi ule, %arg1, %{{.*}} : i128
// CHECK: cf.assert %[[FITS]], "a runtime DMA operand does not fit in 64 bits"
// CHECK: arith.trunci %arg1 : i128 to i64
module {
  aie.device(npu1) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.runtime_sequence @wide(%in: memref<4096xi32>, %n: i128) {
      aiex.npu.dma_memcpy_nd(%in[0, 0, 0, 0][1, 1, %n : i128, 64][0, 0, 64, 1]) {id = 0 : i64, metadata = @a} : memref<4096xi32>
    }
  }
}
