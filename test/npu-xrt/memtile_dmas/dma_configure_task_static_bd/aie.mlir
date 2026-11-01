//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A static memtile BD fills the buffer and a runtime task drains it, so the two
// lowerings must agree on the address of the same memtile buffer.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    %prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons_lock"}

    %buff = aie.buffer(%tile_0_1) {sym_name = "buff"} : memref<4096xi32>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_acq = arith.constant 1 : i32
      aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_acq)
      aie.dma_bd(%buff : memref<4096xi32> offset = 0 len = 4096)
      %c1_rel = arith.constant 1 : i32
      aie.use_lock(%cons_lock, Release, %c1_rel)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>) {
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 4096) {bd_id = 0 : i32}
        aie.end
      }

      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        %c1_acq = arith.constant 1 : i32
        aie.use_lock(%cons_lock, AcquireGreaterEqual, %c1_acq)
        aie.dma_bd(%buff : memref<4096xi32> offset = 0 len = 4096) {bd_id = 8 : i32}
        %c1_rel = arith.constant 1 : i32
        aie.use_lock(%prod_lock, Release, %c1_rel)
        aie.end
      }

      %t2 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<4096xi32> offset = 0 len = 4096) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
    }
  }
}
