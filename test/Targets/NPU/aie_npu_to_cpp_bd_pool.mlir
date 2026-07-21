//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C++ TXN target (aie-npu-to-cpp) for the dynamic BD free-list pool: a
// runtime bd_id lowers to a bd_pool_pop with a nullopt backstop on exhaustion,
// the BD words are written at the runtime pool-derived register address, and
// the id is returned with bd_pool_push. The pool is sized from the target
// model's BD count (getNumBDs) -- 16 on the shim tile here, not a hardcode.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s \
// RUN: | aie-translate --aie-npu-to-cpp | FileCheck %s

// CHECK: inline std::optional<std::vector<uint32_t>> generate_txn_
// The pool is declared once, sized from the target model (shim = 16 BDs).
// CHECK: aie_runtime::BdPool bd_pool_0_0 = aie_runtime::bd_pool_init(16);
// A pop into a fresh variable, failing the build (nullopt) if the pool is empty.
// CHECK: uint32_t bd_{{[0-9]+}}; if (!aie_runtime::bd_pool_pop(bd_pool_0_0, bd_{{[0-9]+}})) return std::nullopt;
// The BD words are emitted as write32s (no constant-address blockwrite).
// CHECK: aie_runtime::txn_append_write32
// The id is returned to the pool.
// CHECK: aie_runtime::bd_pool_push(bd_pool_0_0,

aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.shim_dma_allocation @of_in (%tile_0_0, MM2S, 0)
  aie.runtime_sequence @pool(%in: memref<8192xi32>) {
    %bd = aiex.dma_bd_pool_pop(0, 0) : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) bd_id %bd : i32 {
      aie.dma_bd(%in : memref<8192xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32
  }
}
