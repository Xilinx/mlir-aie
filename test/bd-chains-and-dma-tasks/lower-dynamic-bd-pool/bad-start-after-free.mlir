//===- bad-start-after-free.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// A free pushes the task's id back onto the pool, so a start after it would
// run whatever descriptor the pool's next pop writes into that id.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @start_after_free(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    scf.for %i = %c1 to %n step %c1 {
      %c = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 128)
        aie.end
      }
      aiex.dma_start_task(%c)
      aiex.dma_await_task(%c)
      // expected-note@+1 {{returned here}}
      aiex.dma_free_task(%c)
      // expected-error@+1 {{starts a task whose buffer descriptor ID was already returned to the runtime pool}}
      aiex.dma_start_task(%c)
    }
  }
}
