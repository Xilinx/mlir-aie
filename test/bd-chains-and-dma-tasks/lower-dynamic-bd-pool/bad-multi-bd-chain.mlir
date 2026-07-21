//===- bad-multi-bd-chain.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// v1 supports single-BD tasks only. A multi-BD chain under a runtime loop would
// need runtime next_bd cross-references between the popped ids (not yet
// implemented), so it is diagnosed rather than silently mislowered.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @multi_bd(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    scf.for %i = %c1 to %n step %c1 {
      // expected-error@+1 {{supports single-BD tasks only}}
      %c = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 128)
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 128 len = 128)
        aie.end
      }
      aiex.dma_start_task(%c)
      aiex.dma_await_task(%c)
    }
  }
}
