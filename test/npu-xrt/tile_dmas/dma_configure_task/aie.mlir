//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test demonstrates using aiex.dma_configure_task to configure both
// shim tile and core tile DMAs at runtime with lock synchronization.
//
// Data flow:
//   Host -> Shim MM2S -> Core S2MM -> Buffer -> Core MM2S -> Shim S2MM -> Host

module {
  aie.device(npu1_1col) {
    %shim_tile = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
    %core_tile = aie.tile(0, 2)
    
    %buffer = aie.buffer(%core_tile) {sym_name = "buffer"} : memref<4096xi32>
    
    %prod_lock = aie.lock(%core_tile, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%core_tile, 1) {init = 0 : i32, sym_name = "cons_lock"}
    
    aie.flow(%shim_tile, DMA : 0, %core_tile, DMA : 0)
    aie.flow(%core_tile, DMA : 0, %shim_tile, DMA : 0)
    
    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<4096xi32>) {
      
      %shim_s2mm_task = aiex.dma_configure_task(%shim_tile, S2MM, 0) {
        aie.dma_bd(%arg2 : memref<4096xi32>, 0, 4096) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      
      %shim_mm2s_task = aiex.dma_configure_task(%shim_tile, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32>, 0, 4096) {bd_id = 1 : i32}
        aie.end
      }
      
      // Core tile BDs need to loop (use_next_bd=1) for proper synchronization
      // Using next_bd_id to create self-looping BDs
      %core_s2mm_task = aiex.dma_configure_task(%core_tile, S2MM, 0) {
        aie.use_lock(%prod_lock, Acquire, 1)
        aie.dma_bd(%buffer : memref<4096xi32>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
        aie.use_lock(%cons_lock, Release, 1)
        aie.end
      }
      
      %core_mm2s_task = aiex.dma_configure_task(%core_tile, MM2S, 0) {
        aie.use_lock(%cons_lock, Acquire, 1)
        aie.dma_bd(%buffer : memref<4096xi32>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 1 : i32}
        aie.use_lock(%prod_lock, Release, 1)
        aie.end
      }
      
      aiex.dma_start_task(%shim_s2mm_task)
      aiex.dma_start_task(%core_s2mm_task)
      aiex.dma_start_task(%core_mm2s_task)
      aiex.dma_start_task(%shim_mm2s_task)
      
      aiex.dma_await_task(%shim_s2mm_task)
    }
  }
}
