//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --split-input-file %s \
// RUN:   | FileCheck %s
// RUN: aie-opt --aie-lower-dynamic-bd-pool='enforce-queue-depth=false' \
// RUN:   --verify-diagnostics --split-input-file %s

// This pass is the only one that can guard these pushes: it deliberately keeps
// the loop rolled, so aie-assign-runtime-sequence-bd-ids skips the sequence
// and aie-dma-to-npu only ever sees npu.dma_memcpy_nd. analyzeLoopQueue runs
// the body against the shared queue model until the state repeats, which needs
// no trip count. With enforcement off the same analysis reports instead, which
// is what the second RUN line checks.

// Pushes once per iteration, never awaits. A 4-deep queue is full after four
// iterations, so the fifth push would land on a full one. A single poll in the
// body covers every iteration.
// CHECK-LABEL: @grows
// CHECK:         scf.for
// CHECK:         aiex.npu.maskpoll
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @grows(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %n step %c1 {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      // expected-warning@+1 {{whose task queue is only 4 deep}}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Pushes and retires once per iteration: steady state, nothing to guard.
// CHECK-LABEL: @balanced
// CHECK-NOT:   aiex.npu.maskpoll
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @balanced(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %n step %c1 {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Two pushes and one await per iteration. Counting pushes against awaits makes
// this look like it nets one task a cycle, but the await pops through: it
// retires its own token and the non-token push queued ahead of it, so the body
// drains everything it started. Nothing to guard.
// CHECK-LABEL: @pops_through
// CHECK-NOT:   aiex.npu.maskpoll
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @pops_through(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %n step %c1 {
      %head = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%head)
      %tail = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tail)
      aiex.dma_await_task(%tail)
    }
  }
}

// -----

// A body that pushes more than the whole queue holds fills it inside its very
// first iteration, so every push from the fifth on is guarded.
// CHECK-LABEL: @floods
// CHECK-COUNT-1: aiex.npu.maskpoll
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @floods(%arg0: memref<2048xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    scf.for %i = %c0 to %n step %c1 {
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<2048xi32> offset = 0 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<2048xi32> offset = 256 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<2048xi32> offset = 512 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<2048xi32> offset = 768 len = 256)
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<2048xi32> offset = 1024 len = 256)
        aie.end
      }
      // expected-warning@+1 {{whose task queue is only 4 deep}}
      aiex.dma_start_task(%t4)
    }
  }
}
