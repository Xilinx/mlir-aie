//===- good-runtime-next-bd-memtile.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// The mem tile counterpart of good-runtime-next-bd.mlir, where next_bd is a
// 6-bit field rather than the shim's 4 bits.

// CHECK-LABEL: @runtime_next_bd_memtile
// CHECK: %[[BD0:.*]] = aiex.dma_bd_pool_pop(0, 1) partition [0, 24) : i32
// CHECK: %[[BD1:.*]] = aiex.dma_bd_pool_pop(0, 1) partition [0, 24) : i32
// BD0 word 1: next_bd = BD1 in bits [25:20], use_next_bd (bit 19) set.
// CHECK: %[[C63:.*]] = arith.constant 63 : i32
// CHECK: %[[NB:.*]] = arith.andi %[[BD1]], %[[C63]] : i32
// CHECK: %[[C20:.*]] = arith.constant 20 : i32
// CHECK: %[[NBS:.*]] = arith.shli %[[NB]], %[[C20]] : i32
// CHECK: %[[USE:.*]] = arith.constant 524288 : i32
// CHECK: %[[W1:.*]] = arith.ori %[[NBS]], %[[USE]] : i32
// CHECK: %[[MUL0:.*]] = arith.muli %[[BD0]], %{{.*}} : i32
// CHECK: %[[BASE0:.*]] = arith.addi %{{.*}}, %[[MUL0]] : i32
// CHECK: aiex.npu.blockwrite_values(%[[BASE0]] : i32) values %{{.*}}, %[[W1]], %{{.*}}
// BD1 ends the chain: word 1 is a plain zero.
// CHECK: %[[MUL1:.*]] = arith.muli %[[BD1]], %{{.*}} : i32
// CHECK: %[[BASE1:.*]] = arith.addi %{{.*}}, %[[MUL1]] : i32
// CHECK: aiex.npu.blockwrite_values(%[[BASE1]] : i32) values %{{.*}}, %c0_i32{{[_0-9]*}}, %{{.*}}
// CHECK: aiex.npu.push_queue(0, 1, MM2S : 0) bd_id %[[BD0]]

aie.device(npu2) {
  %tile_0_1 = aie.tile(0, 1)
  %buf = aie.buffer(%tile_0_1) : memref<1024xi32>
  aie.runtime_sequence @runtime_next_bd_memtile() {
    %bd0 = aiex.dma_bd_pool_pop(0, 1) partition [0, 24) : i32
    %bd1 = aiex.dma_bd_pool_pop(0, 1) partition [0, 24) : i32
    %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
      aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 256) bd_id_val %bd0 : i32
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%buf : memref<1024xi32> offset = 256 len = 256) bd_id_val %bd1 : i32
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_bd_pool_push(0, 1) partition [0, 24) bd_id %bd0 : i32
    aiex.dma_bd_pool_push(0, 1) partition [0, 24) bd_id %bd1 : i32
  }
}
