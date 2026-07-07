//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// A BD allocated before a loop, held (unused) across an unrelated loop, and
// freed after the loop is fine: the leak check is per-task (does *this* task's
// handle reach a completion sync?), not per-loop. %t is configured before the
// loop and freed after it, so it is never "configured in a loop that never
// completes it". The in-loop task %u is freed within its own iteration.
// (Answers the review question: an allocation may span a loop it does not
// participate in.)

// CHECK-LABEL: @bd_across_unrelated_loop
// CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 0 : i32}
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @bd_across_unrelated_loop(%arg0: memref<8xi16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // Held across the loop below; id 0 stays reserved the whole time.
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
      aie.end
    }
    aiex.dma_start_task(%t)
    // Unrelated loop: its own task allocates and frees within each iteration,
    // so it must avoid id 0 (taken by %t) and reuse a single id across
    // iterations.
    // CHECK: aie.dma_bd(%arg0 : memref<8xi16>, 0, 8) {bd_id = 1 : i32}
    scf.for %i = %c0 to %c4 step %c1 {
      %u = aiex.dma_configure_task(%tile_0_0, S2MM, 1) {
        aie.dma_bd(%arg0 : memref<8xi16>, 0, 8)
        aie.end
      }
      aiex.dma_start_task(%u)
      aiex.dma_await_task(%u)
    }
    // %t completes only now -- after the loop. Not a leak.
    aiex.dma_await_task(%t)
    aiex.dma_free_task(%t)
  }
}
