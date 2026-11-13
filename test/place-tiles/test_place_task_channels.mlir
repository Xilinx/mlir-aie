//===- test_place_task_channels.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SA
// RUN: not aie-opt --aie-place-tiles %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SEQ

// The runtime sequence configures all six S2MM channels of one mem tile by
// index, with no flow naming them; on a one-column device the objectFIFO link's
// mem tile shares its position and needs a seventh. Both placers count every
// by-index claim the DMA channel analysis makes (program starts, runtime tasks,
// shim allocations), as allocation does ("requires at least 7 S2MM channels").
// SA:  warning: SA placer's resource estimate is over capacity
// SEQ: error: no MemTile on the device has 1 input/1 output DMA channel(s) free

module @task_channels {
  aie.device(npu2_1col) {
    %sh = aie.logical_tile<ShimNOCTile>(?, ?)
    %mt = aie.logical_tile<MemTile>(?, ?)
    %hm = aie.logical_tile<MemTile>(?, ?)
    %c = aie.logical_tile<CoreTile>(?, ?)
    %hb = aie.buffer(%hm) {sym_name = "hand"} : memref<16xi32>
    aie.objectfifo @in(%sh, {%mt}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%mt, {%c}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
    aie.core(%c) {
      %0 = aie.objectfifo.acquire @out(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out(Consume, 1)
      aie.end
    }
    aie.runtime_sequence(%a: memref<16xi32>) {
      %t0 = aiex.dma_configure_task(%hm, S2MM, 0) {
        aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      %t1 = aiex.dma_configure_task(%hm, S2MM, 1) {
        aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      %t2 = aiex.dma_configure_task(%hm, S2MM, 2) {
        aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 2 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      %t3 = aiex.dma_configure_task(%hm, S2MM, 3) {
        aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 3 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
      %t4 = aiex.dma_configure_task(%hm, S2MM, 4) {
        aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 4 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t4)
      aiex.dma_await_task(%t4)
      %t5 = aiex.dma_configure_task(%hm, S2MM, 5) {
        aie.dma_bd(%hb : memref<16xi32> offset = 0 len = 16) {bd_id = 5 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t5)
      aiex.dma_await_task(%t5)
    }
  }
}
