//===- repeat-split-const-operand.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-split-long-repeats %s | FileCheck %s
// RUN: aie-opt --aie-split-long-repeats \
// RUN:   --aie-assign-runtime-sequence-bd-ids='enforce-queue-depth=true' \
// RUN:   --aie-dma-tasks-to-npu %s | FileCheck %s --check-prefix=PUSH

// A start's repeat operand that folds to a constant past the field's maximum
// is split like the attribute form, and the last chunk drops the operand.
// 301 runs = 256 + 45.
// CHECK-LABEL: @const_operand
// CHECK:       aiex.dma_start_task(%[[T:.*]]) {no_token, repeat_count = 255 : i32}
// CHECK-NEXT:  aiex.dma_start_task(%[[T]]) {repeat_count = 44 : i32}
// CHECK-NEXT:  aiex.dma_await_task(%[[T]])
// PUSH-LABEL:  @const_operand
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c255_i32{{(_[0-9]+)?}} {issue_token = false}
// PUSH:        aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %{{.*}} repeat %c44_i32{{(_[0-9]+)?}} {issue_token = true}
// PUSH-NOT:    aiex.npu.push_queue
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @const_operand(%arg0: memref<256xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    %c300 = arith.constant 300 : i32
    aiex.dma_start_task(%t) repeat %c300 : i32
    aiex.dma_await_task(%t)
  }
}
