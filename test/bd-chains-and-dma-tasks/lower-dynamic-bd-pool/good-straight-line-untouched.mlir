//===- good-straight-line-untouched.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s

// A straight-line runtime sequence (no runtime control flow) is left entirely
// to the static allocator: this pass must not touch it, so no pool pop/push
// appears and the configure keeps its constant-attr allocation path.

// CHECK-LABEL: @straight_line
// CHECK-NOT: aiex.dma_bd_pool_pop
// CHECK-NOT: aiex.dma_bd_pool_push
// CHECK: aiex.dma_configure_task
// CHECK-NOT: bd_id %

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @straight_line(%arg0: memref<1024xi32>) {
    %c = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    }
    aiex.dma_start_task(%c)
    aiex.dma_await_task(%c)
  }
}
