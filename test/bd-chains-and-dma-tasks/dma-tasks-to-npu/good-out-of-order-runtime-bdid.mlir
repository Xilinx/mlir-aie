//===- good-out-of-order-runtime-bdid.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An out_of_order S2MM receive BD with a runtime bd_id (dynamic pool) and a
// release-only completion lock, on the shim-NOC dynamic path.

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// CHECK-LABEL: @rt_ooo_dyn

// Enable out-of-order on the shim S2MM channel (CTRL 0x1D200 = 119296, bit 3).
// CHECK: arith.constant 119296 : i32
// CHECK: aiex.npu.maskwrite32(%{{.*}}, %{{.*}}, %{{.*}}) {column = 0 : i32, row = 0 : i32}

// Release lock survives in the packed BD block (word 7 = 0x02040000); dropping
// it would leave 0x02000000 = 33554432.
// CHECK: arith.constant 33816576 : i32
// CHECK: aiex.npu.blockwrite_values

// Out-of-order aliases the TCT bits, so TCT must be disabled.
// CHECK: aiex.npu.push_queue(0, 0, S2MM : 0) bd_id %{{.*}} repeat %{{.*}} {issue_token = false}

aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %lock = aie.lock(%t00, 0) {init = 0 : i32}
  aie.runtime_sequence @rt_ooo_dyn(%arg0: memref<1024xi32>) {
    %c1 = arith.constant 1 : i32
    %bd = aiex.dma_bd_pool_pop(0, 0) : i32
    %t = aiex.dma_configure_task(%t00, S2MM, 0, <pkt_type = 0, pkt_id = 0>) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) bd_id_val %bd : i32
      aie.use_lock(%lock, Release, %c1)
      aie.end
    } {out_of_order}
    aiex.dma_start_task(%t)
  }
}
