//===- good-runtime-next-bd.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-tasks-to-npu %s | FileCheck %s

// A two-BD chain whose ids both come from the dynamic pool. The first BD's
// next_bd field is its successor's runtime id, packed into word 7 next to
// use_next_bd and valid_bd; the last BD ends the chain.

// CHECK-LABEL: @runtime_next_bd_shim
// CHECK: %[[BD0:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// CHECK: %[[BD1:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
// BD0 word 7: next_bd = BD1 in bits [30:27], over valid_bd | use_next_bd.
// CHECK: %[[C15:.*]] = arith.constant 15 : i32
// CHECK: %[[NB:.*]] = arith.andi %[[BD1]], %[[C15]] : i32
// CHECK: %[[C27:.*]] = arith.constant 27 : i32
// CHECK: %[[NBS:.*]] = arith.shli %[[NB]], %[[C27]] : i32
// CHECK: %[[VALID_USE:.*]] = arith.constant 100663296 : i32
// CHECK: %[[W7:.*]] = arith.ori %[[NBS]], %[[VALID_USE]] : i32
// CHECK: %[[MUL0:.*]] = arith.muli %[[BD0]], %{{.*}} : i32
// CHECK: %[[BASE0:.*]] = arith.addi %{{.*}}, %[[MUL0]] : i32
// CHECK: aiex.npu.blockwrite_values(%[[BASE0]] : i32) values {{.*}}, %[[W7]] : i32
// BD1 ends the chain: word 7 is valid_bd alone.
// CHECK: %[[MUL1:.*]] = arith.muli %[[BD1]], %{{.*}} : i32
// CHECK: %[[BASE1:.*]] = arith.addi %{{.*}}, %[[MUL1]] : i32
// CHECK: aiex.npu.blockwrite_values(%[[BASE1]] : i32) values {{.*}}, %c33554432_i32{{[_0-9]*}} : i32
// CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %[[BD0]]
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @runtime_next_bd_shim(%arg0: memref<1024xi32>) {
    %bd0 = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
    %bd1 = aiex.dma_bd_pool_pop(0, 0) partition [0, 16) : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) bd_id_val %bd0 : i32
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256) bd_id_val %bd1 : i32
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %bd0 : i32
    aiex.dma_bd_pool_push(0, 0) partition [0, 16) bd_id %bd1 : i32
  }
}

// -----

// A pinned BD that chains to a pooled one still needs its next_bd at runtime,
// so it takes the packed blockwrite path too, at its constant register base.

// CHECK-LABEL: @pinned_to_runtime_next_bd
// CHECK: %[[BD1:.*]] = aiex.dma_bd_pool_pop(0, 0) partition [1, 16) : i32
// CHECK-NOT: aiex.npu.writebd
// CHECK: %[[NB:.*]] = arith.andi %[[BD1]], %{{.*}} : i32
// CHECK: %[[NBS:.*]] = arith.shli %[[NB]], %{{.*}} : i32
// CHECK: %[[BASE0:.*]] = arith.constant 118784 : i32
// CHECK: aiex.npu.blockwrite_values(%[[BASE0]] : i32)
// CHECK: aiex.npu.blockwrite_values
// CHECK-NOT: aiex.npu.writebd
// CHECK: %[[HEAD:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %[[HEAD]]
aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @pinned_to_runtime_next_bd(%arg0: memref<1024xi32>) {
    %bd1 = aiex.dma_bd_pool_pop(0, 0) partition [1, 16) : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256) {bd_id = 0 : i32}
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 256 len = 256) bd_id_val %bd1 : i32
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_bd_pool_push(0, 0) partition [1, 16) bd_id %bd1 : i32
  }
}
