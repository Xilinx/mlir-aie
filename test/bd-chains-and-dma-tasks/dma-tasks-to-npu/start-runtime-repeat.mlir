//===- start-runtime-repeat.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize --aie-dma-tasks-to-npu %s | FileCheck %s
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize --aie-dma-tasks-to-npu --aie-dma-to-npu %s | FileCheck %s --check-prefix=TXN

// A start's runtime repeat count overrides the task's on its push. The queue
// field is 8 bits wide, so the push is guarded to [0, 255].

// CHECK-LABEL: @replay
// CHECK: scf.for
// CHECK: %[[ID:.*]] = aiex.dma_bd_pool_pop(0, 1)
// CHECK: aiex.npu.push_queue(0, 1, MM2S : 0) bd_id %[[ID]] repeat %{{[a-z0-9_]+}} {issue_token = false}
// CHECK: %[[RC:.*]] = arith.subi %arg1, %{{.*}} : i32
// CHECK: aiex.npu.push_queue(0, 1, MM2S : 0) bd_id %[[ID]] repeat %[[RC]] {issue_token = true}
// CHECK: aiex.dma_bd_pool_push(0, 1) {{.*}} bd_id %[[ID]]

// TXN-LABEL: @replay
// TXN: %[[RC:.*]] = arith.subi %arg1, %{{.*}} : i32
// TXN: %[[WIDE:.*]] = arith.extui %[[RC]] : i32 to i64
// TXN: %[[OK:.*]] = arith.cmpi ule, %[[WIDE]], %c255_i64 : i64
// TXN: cf.assert %[[OK]], "a runtime DMA repeat count exceeds the task queue's [0:255] range (at most 256 executions)"
// TXN: %[[MASKED:.*]] = arith.andi %[[RC]], %{{.*}} : i32
// TXN: arith.shli %[[MASKED]], %{{.*}} : i32
// TXN: aiex.npu.write32
// TXN: aiex.npu.sync
aie.device(npu2) {
  %mt = aie.tile(0, 1)
  %buf = aie.buffer(%mt) {address = 0 : i32} : memref<4096xi32>
  aie.runtime_sequence @replay(%n: index, %uses: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %one = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 {
      %t = aiex.dma_configure_task(%mt, MM2S, 0) {
        aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 4096)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t) {no_token}
      %rc = arith.subi %uses, %one : i32
      aiex.dma_start_task(%t) repeat %rc : i32
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
