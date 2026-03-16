//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Core tile buffers
    %in_buff = aie.buffer(%tile_0_2) {sym_name = "in_buff"} : memref<1024xi32>

    // Locks
    %prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "cons_lock"}

    // Flows: shim -> memtile -> core -> memtile -> shim
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)

    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      // Configure shim DMA to send data to core
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.end
      }

      // Configure core DMA to receive from shim
      %t3 = aiex.dma_configure_task(%tile_0_2, S2MM, 0) {
        aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%in_buff : memref<1024xi32>, 0, 1024) {bd_id = 0 : i32}
        aie.use_lock(%cons_lock, Release, 1)
        aie.end
      }

      // Start input path
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t3)

      // Configure core DMA to send to memtile
      %t4 = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
        aie.use_lock(%cons_lock, AcquireGreaterEqual, 1)
        aie.dma_bd(%in_buff : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.use_lock(%prod_lock, Release, 1)
        aie.end
      }

      // Configure shim DMA to receive from memtile
      %t7 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<1024xi32>, 0, 1024) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      // Start output path
      aiex.dma_start_task(%t4)
      aiex.dma_start_task(%t7)
      aiex.dma_await_task(%t7)
    }
  }
}
