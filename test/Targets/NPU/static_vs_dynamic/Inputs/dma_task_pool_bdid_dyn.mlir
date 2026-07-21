//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic-pool companion input for dma_task_pool_bdid.mlir (not a standalone
// test -- lives under Inputs/ so lit ignores it). The BD id is drawn at runtime
// from the free-list pool; a fresh pool pops 0, matching the static oracle's
// pinned bd_id = 0.
//
// %unused is an i32 arg so the generated function matches the comparator's
// DYN_FN(ARGVAL) call signature; the pool id does not depend on it.
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)
    aie.runtime_sequence @pool_dynamic(%in: memref<1024xi32>, %unused: i32) {
      %bd = aiex.dma_bd_pool_pop(0, 0) : i32
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) bd_id %bd : i32 {
        aie.dma_bd(%in : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32
    }
  }
}
