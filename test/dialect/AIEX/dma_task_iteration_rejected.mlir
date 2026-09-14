//===- dma_task_iteration_rejected.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A runtime-valued BD still rejects the #aie.bd_iteration attribute: the
// dynamic BD-word encoder does not implement it. A compile-time-constant BD
// does not hit this (see good-iteration-attribute.mlir in
// bd-chains-and-dma-tasks/dma-tasks-to-npu/); on that path iteration is
// expressed via the outermost sizes/strides dimension only when a runtime
// value forces it here.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>, %n: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{the iteration attribute requires a compile-time-constant buffer descriptor on the runtime-sequence path}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, %n, 8, 4] strides = [4096, 512, 4, 1]) { bd_id = 5 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
        aie.end
      }
    }
  }
}
