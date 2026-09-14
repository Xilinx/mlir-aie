//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics \
// RUN:         --split-input-file %s

// The straight-line queue check cannot run on this path: the loop stays rolled,
// so there is no finite push sequence to count. analyzeLoopQueue runs the body
// against the shared queue model until the state repeats, which needs no trip
// count and reports the iteration the queue actually fills on.

// Pushes once per iteration, never awaits. A 4-deep queue is full after four
// iterations, so it is the fifth push that lands on a full queue.
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
      // expected-warning@+1 {{this loop fills the 4-deep DMA task queue on tile (0,0) MM2S channel 0: on iteration 5 this push lands on a full queue}}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// Pushes and retires once per iteration: steady state, no growth.
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
// drains everything it started and occupancy never grows. Must stay quiet.
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

// A body that pushes more than the whole queue holds overflows inside its very
// first iteration, so the reported iteration is 1 and not the zero a
// depth-over-delta division would give.
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
      // expected-warning@+1 {{on iteration 1 this push lands on a full queue}}
      aiex.dma_start_task(%t4)
    }
  }
}
