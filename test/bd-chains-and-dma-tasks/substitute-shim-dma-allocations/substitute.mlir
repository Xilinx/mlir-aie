//===- substitute.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-substitute-shim-dma-allocations %s | FileCheck %s

// Each task names its allocation's tile, direction and channel; the body moves
// with it.

// CHECK: %[[T:.*]] = aie.tile(1, 0)
// CHECK: @two_tasks
// CHECK: aiex.dma_configure_task(%[[T]], MM2S, 1) {
// CHECK:   aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
// CHECK: } {issue_token = true}
// CHECK: aiex.dma_configure_task(%[[T]], S2MM, 0) {
// CHECK:   aie.dma_bd(%arg1 : memref<64xi32> offset = 0 len = 64) {bd_id = 1 : i32}
// CHECK: }
// CHECK-NOT: dma_configure_task_for
module {
  aie.device(npu2) {
    %t = aie.tile(1, 0)
    aie.shim_dma_allocation @in(%t, MM2S, 1)
    aie.shim_dma_allocation @out(%t, S2MM, 0)
    aie.runtime_sequence @two_tasks(%a: memref<64xi32>, %b: memref<64xi32>) {
      %t0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      %t1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%b : memref<64xi32> offset = 0 len = 64) {bd_id = 1 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t0)
    }
  }
}

// -----

// A task whose symbol names no shim DMA allocation fails the pass.

module {
  aie.device(npu2) {
    %t = aie.tile(1, 0)
    %c = aie.tile(1, 2)
    aie.objectfifo @not_an_alloc(%t, {%c}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.shim_dma_allocation @in(%t, MM2S, 0)
    aie.runtime_sequence @bad(%a: memref<64xi32>) {
      // expected-error@+1 {{no shim DMA allocation found for symbol}}
      %t0 = aiex.dma_configure_task_for @not_an_alloc {
        aie.dma_bd(%a : memref<64xi32> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
    }
  }
}
