//===- aie_npu_to_cpp_bd_pool_handwritten_pop_copy_reserve.mlir -*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A sequence with a hand-written aiex.dma_bd_pool_pop (as
// npu-xrt/dma_s2mm_ooo_shim_recv writes one) reaches neither of
// aie-lower-dynamic-bd-pool's own lowering paths -- it has no runtime
// control flow for the pass to lower, and it already carries a pop -- so the
// pass used to skip it, and a copy of BD 0 on the same tile shared it with
// the hand-written pop's first pop. The copy's id is still withheld.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool %s | FileCheck %s --check-prefix=POOL
// RUN: aie-opt --aie-lower-dynamic-bd-pool --canonicalize \
// RUN:   --aie-dma-tasks-to-npu --aie-dma-to-npu %s \
// RUN: | aie-translate --aie-npu-to-cpp | FileCheck %s

// POOL: aie.runtime_sequence @handwritten
// POOL-NEXT: aiex.dma_bd_pool_pop(0, 0) partition [0, 16) id 0 : i32

// CHECK: aie_runtime::BdPool bd_pool_0_0_0_16 = aie_runtime::bd_pool_init_range(0, 16);
// CHECK-NOT: bd_pool_pop(
// CHECK: aie_runtime::bd_pool_reserve(bd_pool_0_0_0_16, 0);
// CHECK: bd_pool_pop(bd_pool_0_0_0_16,

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl0(%t00, MM2S, 0)
    aie.runtime_sequence @handwritten(%arg0: memref<16xi32>, %pkts : memref<?xi32>) {
      %c0i = arith.constant 0 : i32
      %c1i = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl0} : memref<?xi32>
      %bda = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
      %t = aiex.dma_configure_task(%t00, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16) bd_id_val %bda : i32
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %bda : i32
      aiex.npu.sync(%c0i, %c0i, %c1i, %c0i, %c1i, %c1i) : i32, i32, i32, i32, i32, i32
    }
  }
}
