//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics \
// RUN:         --split-input-file %s

// The straight-line queue check cannot run on this path: the loop stays rolled,
// so there is no finite push sequence to count. The per-iteration delta is
// enough by itself and needs no trip count -- a body that pushes more onto a
// channel than it retires grows occupancy every iteration.

// Pushes once per iteration, never awaits: unbounded.
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
      // expected-warning@+1 {{each iteration of this loop nets 1 outstanding DMA task(s) on tile (0,0) MM2S channel 0}}
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
