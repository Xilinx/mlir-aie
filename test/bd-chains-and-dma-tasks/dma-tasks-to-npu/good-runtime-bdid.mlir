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
// bd_id, so it becomes 118784 + bd_id*32 -- so WriteBdToBlockWritePattern's
// constant-address blockwrite cannot be formed.
//
// Instead the whole 8-word register block is packed into ONE
// npu.blockwrite_values at that runtime base, whose payload mixes the constant
// template words (buffer_offset word 1, packet word 2, valid/lock word 7) with
// the encoder's runtime size/stride words -- so no per-word write32 remains for
// this BD. Emitting a real block-write is required for aiebu: it folds a
// following address_patch instruction into an ELF relocation on the 
// blockwrite.
//
// The buffer address_patch then takes a runtime addr operand pointing INSIDE
// that range (base + 4, word 1), and the queue push uses the runtime bd_id.

// CHECK-LABEL: @runtime_bdid
// The runtime BD register base: 118784 + bd_id*32.
// CHECK: %[[POP:.*]] = aiex.dma_bd_pool_pop(0, 0) : i32
// CHECK: %[[MUL:.*]] = arith.muli %[[POP]], %{{.*}} : i32
// CHECK: %[[BASE:.*]] = arith.addi %{{.*}}, %[[MUL]] : i32
// One packed blockwrite carries all 8 words; no write32 configures this BD.
// CHECK: aiex.npu.blockwrite_values(%[[BASE]] : i32) values
// CHECK-NOT: aiex.npu.write32
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
