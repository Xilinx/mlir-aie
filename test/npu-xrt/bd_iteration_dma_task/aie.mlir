//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A single S2MM buffer descriptor on the runtime-sequence path
// (aiex.dma_configure_task) receives a 4096-element stream in four chunks; its
// #aie.bd_iteration advances the write base one slot per run (current=2), so
// chunk k lands at slot ((2 + k) % 4). Drained back to the host to check.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    %prod_lock = aie.lock(%tile_0_1, 0) {init = 4 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons_lock"}
    %out_buff = aie.buffer(%tile_0_1) {sym_name = "out_buff"} : memref<4096xi32>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 4096) {bd_id = 0 : i32}
        aie.end
      }

      // The iteration BD under test; repeat_count=3 gives it four executions,
      // one cons_lock release each.
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1)
        aie.dma_bd(%out_buff : memref<4096xi32> offset = 0 len = 1024) {bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 4, stride = 1024, current = 2>}
        %c1_0 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_0)
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)

      %t2 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        %c4 = arith.constant 4 : i32
        aie.use_lock(%cons_lock, AcquireGreaterEqual, %c4)
        aie.dma_bd(%out_buff : memref<4096xi32> offset = 0 len = 4096) {bd_id = 1 : i32}
        %c4_0 = arith.constant 4 : i32
        aie.use_lock(%prod_lock, Release, %c4_0)
        aie.end
      }
      %t3 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 4096) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
    }
  }
}
