//===- good-out-of-order-id-dynamic.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s
// A sender BD with a runtime bd_id lowers through the dynamic path to one
// blockwrite instead of an npu.writebd, so its out_of_order_id rides word[2]
// (bits [29:24]) rather than a writebd attribute. Id 5 with packet pkt_id 1
// gives word[2] = enable_packet | (5 << 24) | (1 << 19) = 0x45080000 =
// 1158152192, the third value of the blockwrite.
// CHECK-LABEL: @dyn_ooo
// CHECK: %[[W2:.*]] = arith.constant 1158152192 : i32
// CHECK: aiex.npu.blockwrite_values({{.*}}) values {{.*}}, %[[W2]], {{.*}}
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence @dyn_ooo(%arg0: memref<1024xi32>) {
      %bd = aiex.dma_bd_pool_pop(0, 0) : i32
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) bd_id %bd : i32 {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {out_of_order_id = 5 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32
    }
  }
}
