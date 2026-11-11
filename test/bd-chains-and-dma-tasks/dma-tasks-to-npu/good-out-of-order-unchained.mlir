//===- good-out-of-order-unchained.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An out-of-order S2MM channel ignores Next_BD; the BD chain of an
// out_of_order task only lists its receive BDs. Each receive BD is written
// unchained (use_next_bd = 0), as the static AIE-RT lowering writes it.

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

// CHECK-LABEL: @ooo_chain_static_ids
// CHECK: aiex.npu.writebd
// CHECK-SAME: bd_id = 2
// CHECK-SAME: next_bd = 0
// CHECK-SAME: use_next_bd = 0
// CHECK: aiex.npu.writebd
// CHECK-SAME: bd_id = 5
// CHECK-SAME: next_bd = 0
// CHECK-SAME: use_next_bd = 0
module @ooo_chain_static_ids {
  aie.device(npu2) {
    %t01 = aie.tile(0, 1)
    %b = aie.buffer(%t01) { address = 0 : i32 } : memref<16xi32>
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %t = aiex.dma_configure_task(%t01, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%b : memref<16xi32> offset = 0 len = 8) {bd_id = 2 : i32}
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%b : memref<16xi32> offset = 8 len = 8) {bd_id = 5 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// With runtime BD ids a chained BD packs the next BD id into word 7 with a
// shift (arith.shli); an unchained BD's word 7 is a constant.
// CHECK-LABEL: @ooo_chain_runtime_ids
// CHECK-NOT: arith.shli
// CHECK: aiex.npu.blockwrite_values
// CHECK-NOT: arith.shli
// CHECK: aiex.npu.blockwrite_values
// CHECK-NOT: arith.shli
// CHECK: aiex.npu.push_queue
module @ooo_chain_runtime_ids {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %bd0 = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
      %bd1 = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
      %t = aiex.dma_configure_task(%t00, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) bd_id_val %bd0 : i32
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256) bd_id_val %bd1 : i32
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}
