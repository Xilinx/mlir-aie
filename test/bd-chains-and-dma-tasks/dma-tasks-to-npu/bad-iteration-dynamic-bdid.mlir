//===- bad-iteration-dynamic-bdid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A compile-time-constant BD (see good-iteration-attribute.mlir) still
// rejects #aie.bd_iteration when the BD draws its bd_id from the runtime
// free-list pool (good-runtime-bdid.mlir): the verifier cannot see this from
// the sizes/strides alone, so the dynamic BD-word encoder rejects it once
// dma-tasks-to-npu resolves that the bd_id is runtime.

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %bd = aiex.dma_bd_pool_pop(0, 0) : i32
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{the iteration attribute is not supported with a dynamic (runtime-pool) bd_id}}
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) bd_id_val %bd : i32 {iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32
    }
  }
}
