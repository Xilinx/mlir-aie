//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-substitute-shim-dma-allocations \
// RUN:         --aie-assign-runtime-sequence-bd-ids \
// RUN:         --split-input-file %s | FileCheck %s

// Two aie.tile ops for one (0, 0) (accepted, see
// test/Passes/assign-bd-ids/runtime_vs_static_multi_tile.mlir): the copy's
// allocation names %s0, the task is configured on %s1. The task is in
// flight when the copy programs BD 0 -- the same physical tile, so this is
// the same clash as @one_tile_op below, and the allocator must reallocate
// the task off id 0 either way, not treat %s0 and %s1 as unrelated tiles.

// CHECK-LABEL: @duplicate_tile_ops
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd({{.*}}) {id = 0 : i64
module @duplicate_tile_ops {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%s0, MM2S, 0)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %t = aiex.dma_configure_task(%s1, MM2S, 1) {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// Control: one tile op, same clash.
// CHECK-LABEL: @one_tile_op
// CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd({{.*}}) {id = 0 : i64
module @one_tile_op {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%s0, MM2S, 0)
    aie.runtime_sequence @run(%x : memref<16xi32>, %pkts : memref<?xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %t = aiex.dma_configure_task(%s0, MM2S, 1) {
        aie.dma_bd(%x : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.npu.dma_memcpy_nd(%pkts[0, 0, 0, 0][1, 1, 1, 3][0, 0, 0, 1]) {id = 0 : i64, issue_token = true, metadata = @ctrl} : memref<?xi32>
      aiex.npu.sync(%c0, %c0, %c1, %c0, %c1, %c1) : i32, i32, i32, i32, i32, i32
      aiex.dma_await_task(%t)
    }
  }
}
