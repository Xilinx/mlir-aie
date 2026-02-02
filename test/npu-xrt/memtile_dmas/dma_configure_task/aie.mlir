//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test demonstrates using aiex.dma_configure_task to configure memtile
// DMAs at runtime. Functionally equivalent to the writebd test.
//
// Data flow:
//   Host -> Shim MM2S -> Memtile S2MM -> Buffer -> Memtile MM2S -> Shim S2MM -> Host

module {
  aie.device(npu1_1col) {
    %shim_tile = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %mem_tile = aie.tile(0, 1)
    
    // Local buffer in the memtile - address 0
    %buffer = aie.buffer(%mem_tile) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "buffer"} : memref<4096xi32>
    
    // Locks for synchronization between S2MM and MM2S on memtile
    %prod_lock = aie.lock(%mem_tile, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%mem_tile, 1) {init = 0 : i32, sym_name = "cons_lock"}
    
    aie.flow(%shim_tile, DMA : 0, %mem_tile, DMA : 0)
    aie.flow(%mem_tile, DMA : 0, %shim_tile, DMA : 0)
    
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      
      // Configure shim tile S2MM to receive output data
      %shim_s2mm_task = aiex.dma_configure_task(%shim_tile, S2MM, 0) {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      
      // Configure shim tile MM2S to send input data
      %shim_mm2s_task = aiex.dma_configure_task(%shim_tile, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096) {bd_id = 1 : i32}
        aie.end
      }
      
      // Configure memtile S2MM to receive data from shim
      // Uses locks for synchronization, loops back to itself
      %mem_s2mm_task = aiex.dma_configure_task(%mem_tile, S2MM, 0) {
        aie.use_lock(%prod_lock, Acquire, 1)
        aie.dma_bd(%buffer : memref<4096xi32>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.use_lock(%cons_lock, Release, 1)
        aie.end
      }
      
      // Configure memtile MM2S to send data to shim
      // Uses locks for synchronization, loops back to itself
      %mem_mm2s_task = aiex.dma_configure_task(%mem_tile, MM2S, 0) {
        aie.use_lock(%cons_lock, Acquire, 1)
        aie.dma_bd(%buffer : memref<4096xi32>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
        aie.use_lock(%prod_lock, Release, 1)
        aie.end
      }
      
      // Start all DMA tasks
      aiex.dma_start_task(%shim_s2mm_task)
      aiex.dma_start_task(%mem_s2mm_task)
      aiex.dma_start_task(%mem_mm2s_task)
      aiex.dma_start_task(%shim_mm2s_task)
      
      // Wait for output to complete
      aiex.dma_await_task(%shim_s2mm_task)
    }
  }
}
