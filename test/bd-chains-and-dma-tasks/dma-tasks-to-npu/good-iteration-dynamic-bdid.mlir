//===- good-iteration-dynamic-bdid.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A BD that draws its bd_id from the runtime free-list pool goes through the
// dynamic BD-word encoder, which packs #aie.bd_iteration into shim word 6:
// current 2 at bit 26, wrap 3 at bit 20, stride 15 (0x0830000F).

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// CHECK: %[[W6:.*]] = arith.constant 137363471 : i32
// CHECK: aiex.npu.blockwrite_values(%{{.*}} : i32) values %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[W6]], %{{.*}} : i32, i32, i32, i32, i32, i32, i32, i32

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      %bd = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) bd_id_val %bd : i32 {iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>}
        aie.end
      } {issue_token = true, repeat_count = 3 : i32}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %bd : i32
    }
  }
}
