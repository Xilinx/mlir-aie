//===- dma_task_iteration_rejected.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The grouped #aie.bd_iteration attribute drives the structural path.
// On the runtime-sequence path, iteration is expressed via the outermost
// sizes/strides dimension.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{the iteration attribute is not supported on the runtime-sequence path}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64) { iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
        aie.end
      }
    }
  }
}
