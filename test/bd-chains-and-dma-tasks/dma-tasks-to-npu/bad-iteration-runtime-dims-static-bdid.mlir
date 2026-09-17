//===- bad-iteration-runtime-dims-static-bdid.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// verifyTaskBDDimensions normally catches a runtime-valued BD with
// #aie.bd_iteration before this pass ever runs (dma_task_iteration_rejected.mlir),
// so the lowering-path diagnostic below is unreachable with a placed tile and
// verification enabled. Disable both to isolate the pass's own diagnostic:
// with a STATIC bd_id but a runtime-valued dimension forcing the dynamic BD
// path, it must report the compile-time-constant BD cause, not the dynamic
// (runtime-pool) bd_id cause the same branch also reports for a different
// input (bad-iteration-dynamic-bdid.mlir).

// RUN: aie-opt --mlir-very-unsafe-disable-verifier-on-parsing --verify-each=false --verify-diagnostics --aie-dma-tasks-to-npu %s

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<64xi32>, %n: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error @+1 {{the iteration attribute requires a compile-time-constant buffer descriptor on the runtime-sequence path}}
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 64 sizes = [1, %n, 8, 4] strides = [4096, 512, 4, 1]) { bd_id = 5 : i32, iteration = #aie.bd_iteration<size = 4, stride = 16, current = 0> }
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
    }
  }
}
