//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids='warn-unsafe-bd-reuse=true warn-unreleased-transfers=false' \
// RUN:         %s -o /dev/null 2>&1 | FileCheck %s

// Two distinct, unrelated frees of BD 0 whose reuse both land on one explicit
// location (as aie-ctrl-packet-to-dma's generated copies all did before they
// carried their own control packet's location): the merged warning keeps
// every free it absorbed, not just the first.

// CHECK: reuses buffer descriptor ID 0
// CHECK-COUNT-2: note: released here
// CHECK-NOT: note: released here
module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @in(%shim, MM2S, 0)
    aie.runtime_sequence @run(%x : memref<16xi32>) {
      %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32} loc("shared_reuse")
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%t1)
      aiex.dma_free_task(%t1)
      %t2 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32} loc("shared_reuse")
        aie.end
      } {alloc = @in}
      aiex.dma_start_task(%t2)
    }
  }
}
