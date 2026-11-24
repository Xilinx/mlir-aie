//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A runtime task pads a 4x16 i8 tile held in a memtile to 6x28: one row above
// and below, 4 bytes before and 8 after each row. i8 pads count elements in the
// IR and 32-bit words in the BD, and the BD length counts the padding.

module {
  aie.device(NPUDEVICE) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)

    %prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "prod_lock"}
    %cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "cons_lock"}

    %buff = aie.buffer(%tile_0_1) {sym_name = "buff"} : memref<64xi8>

    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:
      %c1_acq = arith.constant 1 : i32
      aie.use_lock(%prod_lock, AcquireGreaterEqual, %c1_acq)
      aie.dma_bd(%buff : memref<64xi8> offset = 0 len = 64)
      %c1_rel = arith.constant 1 : i32
      aie.use_lock(%cons_lock, Release, %c1_rel)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<64xi8>, %arg1: memref<168xi8>) {
      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi8> offset = 0 len = 64) {bd_id = 0 : i32}
        aie.end
      }

      %t1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        %c1_acq = arith.constant 1 : i32
        aie.use_lock(%cons_lock, AcquireGreaterEqual, %c1_acq)
        aie.dma_bd(%buff : memref<64xi8> offset = 0 len = 168 sizes = [4, 16] strides = [16, 1] pad [<const_pad_before = 1, const_pad_after = 1>, <const_pad_before = 4, const_pad_after = 8>]) {bd_id = 8 : i32}
        %c1_rel = arith.constant 1 : i32
        aie.use_lock(%prod_lock, Release, %c1_rel)
        aie.end
      }

      %t2 = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {
        aie.dma_bd(%arg1 : memref<168xi8> offset = 0 len = 168) {bd_id = 1 : i32}
        aie.end
      } {issue_token = true}

      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
    }
  }
}
