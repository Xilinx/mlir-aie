//===- good-runtime-bdid.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A dma_configure_task carrying a RUNTIME bd_id (the dynamic free-list pool's
// dma_bd_pool_pop result, on the configure's bd_id_val operand). The BD register
// block address is then runtime -- getDmaBdAddress(col,row,bd_id) is linear in
// bd_id, so it becomes 118784 + bd_id*32 -- and the whole BD cannot be a
// constant-address blockwrite. Instead every word is a write32 at that runtime
// base: the template words (buffer_offset word 1, packet word 2, valid/lock
// word 7) plus the size/stride words from the encoder, and the buffer
// address_patch takes a runtime addr operand. The queue push uses the runtime
// bd_id directly.

// CHECK-LABEL: @runtime_bdid
// The runtime BD register base: 118784 + bd_id*32.
// CHECK: %[[POP:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK: %[[MUL:.*]] = arith.muli %[[POP]], %{{.*}} : i32
// CHECK: %[[BASE:.*]] = arith.addi %{{.*}}, %[[MUL]] : i32
// Template + size/stride words are write32s at that runtime base (not a
// constant-address blockwrite).
// CHECK-NOT: aiex.npu.blockwrite
// CHECK: aiex.npu.write32
// The buffer pointer patch targets the runtime register address.
// CHECK: aiex.npu.address_patch(%{{.*}} : i32) addr %{{.*}} : i32
// The queue push launches the runtime bd_id.
// CHECK: aiex.npu.push_queue(0, 0, MM2S : 0) bd_id %[[POP]]

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @runtime_bdid(%arg0: memref<1024xi32>) {
    %bd = aiex.dma_bd_pool_pop(0, 0) : i32
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) bd_id %bd : i32 {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 256)
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_bd_pool_push(0, 0) bd_id %bd : i32
  }
}
