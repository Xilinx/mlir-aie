//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The C++ TXN target for a ROLLED runtime-bound loop: the scf.for survives to
// aie-npu-to-cpp and becomes an emitc.for (a real C++ for-loop), with the BD
// free-list pool pop/push inside it and a RUNTIME op-count (__opcount) fed to
// the header -- the loop body appends a runtime number of ops. This is the
// end-to-end dynamic path the static allocator rejects.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s \
// RUN: | aie-translate --aie-npu-to-cpp | FileCheck %s

// The pool is declared once from the target model's BD count.
// CHECK: aie_runtime::BdPool bd_pool_0_0 = aie_runtime::bd_pool_init(16);
// The op-count is a runtime accumulator, not a compile-time literal.
// CHECK: uint32_t __opcount = 0;
// The prologue pops the initial id.
// CHECK: uint32_t bd_{{[0-9]+}}; if (!aie_runtime::bd_pool_pop(bd_pool_0_0, bd_{{[0-9]+}})) return std::nullopt;
// The runtime-bound loop is a real C++ for-loop over the trip-count param.
// CHECK: for (size_t
// Inside the loop: pop, count, and push the previous id.
// CHECK: uint32_t bd_{{[0-9]+}}; if (!aie_runtime::bd_pool_pop(bd_pool_0_0, bd_{{[0-9]+}})) return std::nullopt;
// CHECK: ++__opcount;
// CHECK: aie_runtime::bd_pool_push(bd_pool_0_0,
// The header reads the runtime count.
// CHECK: aie_runtime::txn_prepend_header(txn, __opcount,

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @rolled(%arg0: memref<1024xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_free_task(%prev)
      scf.yield %t : index
    }
    aiex.dma_await_task(%last)
    aiex.dma_free_task(%last)
  }
}
