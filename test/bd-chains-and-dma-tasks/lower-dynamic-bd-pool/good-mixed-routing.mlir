//===- good-mixed-routing.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The aiecc pass order: the dynamic pool pass claims runtime-bound sequences,
// then the static allocator handles the rest and SKIPS the pooled ones. A
// straight-line sequence and a runtime-bound sequence in the same device each
// take the correct path with no conflict.

// RUN: aie-opt --aie-unroll-runtime-sequence-loops --canonicalize \
// RUN:   --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// The straight-line sequence: static allocator assigns a constant bd_id (no
// pool ops).
// CHECK-LABEL: @straight
// CHECK-NOT: aiex.dma_bd_pool_pop
// CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}

// The runtime-bound sequence: kept rolled with pool pop/push, no static id.
// CHECK-LABEL: @dynamic
// CHECK: aiex.dma_bd_pool_pop
// CHECK: scf.for
// CHECK: aiex.dma_bd_pool_push

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @straight(%arg0: memref<1024xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
  }
  aie.runtime_sequence @dynamic(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
