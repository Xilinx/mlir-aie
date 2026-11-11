//===- aie_npu_to_cpp_bd_pool_copy_reserve.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An aiex.npu.dma_memcpy_nd programs its BD id on its tile outside the runtime
// pool, as a static BD does. Here a copy of BD 0 on MM2S 0 sits next to a
// runtime-bound ping-pong on MM2S 1 that draws from the pool of tile (0, 0),
// so the pool withholds 0 and its first pop returns something else.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s --check-prefix=POOL
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s \
// RUN: | aie-translate --aie-npu-to-cpp | FileCheck %s

// POOL: aie.runtime_sequence @pool_over_copy
// POOL-NEXT: aiex.dma_bd_pool_pop(0, 0) partition [0, 16) id 0 : i32

// CHECK: aie_runtime::BdPool bd_pool_0_0_0_16 = aie_runtime::bd_pool_init_range(0, 16);
// CHECK-NOT: bd_pool_pop(
// CHECK: aie_runtime::bd_pool_reserve(bd_pool_0_0_0_16, 0);
// CHECK: bd_pool_pop(bd_pool_0_0_0_16,

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%tile_0_0, MM2S, 0)
    aie.runtime_sequence @pool_over_copy(%arg0: memref<1024xi32>, %pkts : memref<?xi32>, %n: index) {
      %c1 = arith.constant 1 : index
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      %init = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%init)
      %last = scf.for %i = %c1 to %n step %c1 iter_args(%prev = %init) -> (index) {
        %t = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
          aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
          aie.end
        } {issue_token = true}
        aiex.dma_start_task(%t)
        aiex.dma_await_task(%prev)
        aiex.dma_free_task(%prev)
        scf.yield %t : index
      }
      aiex.dma_await_task(%last)
      aiex.dma_free_task(%last)
      %c0 = arith.constant 0 : i32
      %c1i = arith.constant 1 : i32
      aiex.npu.sync(%c0, %c0, %c1i, %c0, %c1i, %c1i) : i32, i32, i32, i32, i32, i32
    }
  }
}
