//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    
    // Locks for synchronization between S2MM and MM2S on memtile
    %prod_lock = aie.lock(%tile_0_1, 0) {init = 4 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons_lock"}

    %in_buff = aie.buffer(%tile_0_1) {sym_name = "in_buff"} : memref<4096xi32>
    %out_buff = aie.buffer(%tile_0_1) {sym_name = "out_buff"} : memref<4096xi32>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      // Configure shim DMA to send data to memtile
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 4096) {bd_id = 0 : i32}
        aie.end
      }

      // Configure memtile DMA to receive from shim
      %t1 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        %c1_ul1 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul1)
        aie.dma_bd(%out_buff : memref<4096xi32> offset = 1024 len = 1024) {bd_id = 0 : i32}
        %c1_ul2 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul2)
        aie.next_bd ^bb1
      ^bb1:
        %c1_ul3 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul3)
        aie.dma_bd(%out_buff : memref<4096xi32> offset = 3072 len = 1024) {bd_id = 1 : i32}
        %c1_ul4 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul4)
        aie.next_bd ^bb2
      ^bb2:
        %c1_ul5 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul5)
        aie.dma_bd(%out_buff : memref<4096xi32> offset = 0 len = 1024) {bd_id = 2 : i32}
        %c1_ul6 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul6)
        aie.next_bd ^bb3
      ^bb3:
        %c1_ul7 = arith.constant 1 : i32
        aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_ul7)
        aie.dma_bd(%out_buff : memref<4096xi32> offset = 2048 len = 1024) {bd_id = 3 : i32}
        %c1_ul8 = arith.constant 1 : i32
        aie.use_lock(%cons_lock, Release, %c1_ul8)
        aie.end
      }

      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)

      // Configure memtile DMA to send data to shim
      %t2 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        %c4_ul9 = arith.constant 4 : i32
        aie.use_lock(%cons_lock, AcquireGreaterEqual, %c4_ul9)
        aie.dma_bd(%out_buff : memref<4096xi32> offset = 0 len = 4096) {bd_id = 4 : i32}
        %c4_ul10 = arith.constant 4 : i32
        aie.use_lock(%prod_lock, Release, %c4_ul10)
        aie.end
      }

      // Configure shim DMA to receive data from memtile
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
