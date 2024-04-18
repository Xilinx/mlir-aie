//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    %buffer_0_2 = aie.buffer(%tile_0_2) : memref<16xi32>
    %buffer_0_2_1 = aie.buffer(%tile_0_2) : memref<16xi32>

    %lock_0_2_0 = aie.lock(%tile_0_2, 0) {init = 1 : i32}
    %lock_0_2_1 = aie.lock(%tile_0_2, 1) {init = 0 : i32}
    %lock_0_2_2 = aie.lock(%tile_0_2, 2) {init = 1 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 3) {init = 0 : i32}

    // forward
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    // backward
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)

    %core_0_2 = aie.core(%tile_0_2) {
      // aie-core-to-standard somehow erases this if it's outside
      // the core op
      %c256_i32 = arith.constant 256 : i32
      affine.for %arg0 = 0 to 4 step 1 {
        aie.use_lock(%lock_0_2_1, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_2, AcquireGreaterEqual, 1)
        affine.for %arg1 = 0 to 16 step 1 {
          %0 = memref.load %buffer_0_2[%arg1] : memref<16xi32>
          %1 = arith.addi %0, %c256_i32: i32
          memref.store %1, %buffer_0_2_1[%arg1] : memref<16xi32>
        }
        aie.use_lock(%lock_0_2_0, Release, 1)
        aie.use_lock(%lock_0_2_3, Release, 1)
      }
      aie.end
    }

    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %buffer_0_1 = aie.buffer(%tile_0_1) : memref<16xi32>
      %buffer_0_1_0 = aie.buffer(%tile_0_1) : memref<16xi32>
      %lock_0_1_0 = aie.lock(%tile_0_1, 0) {init = 1 : i32}
      %lock_0_1_1 = aie.lock(%tile_0_1, 1) {init = 0 : i32}
      %lock_0_1_2 = aie.lock(%tile_0_1, 2) {init = 1 : i32}
      %lock_0_1_3 = aie.lock(%tile_0_1, 3) {init = 0 : i32}

      // forward
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_0_1 : memref<16xi32>)
        aie.use_lock(%lock_0_1_1, Release, 1)
      }]
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_0_1 : memref<16xi32>)
        aie.use_lock(%lock_0_1_0, Release, 1)
      }]
      // backward
      %2 = aie.dma(S2MM, 1) [{
        aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_0_1_0 : memref<16xi32>)
        aie.use_lock(%lock_0_1_3, Release, 1)
      }]
      %3 = aie.dma(MM2S, 1) [{
        aie.use_lock(%lock_0_1_3, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_0_1_0 : memref<16xi32>)
        aie.use_lock(%lock_0_1_2, Release, 1)
      }]
      aie.end
    }
    // streaming interface synchronizes these two tiles
    %mem_0_2 = aie.mem(%tile_0_2) {
     // in
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_2_0, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_0_2 : memref<16xi32>)
        aie.use_lock(%lock_0_2_1, Release, 1)
      }]
      // out
      %1 = aie.dma(MM2S, 0) [{
        aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
        aie.dma_bd(%buffer_0_2_1 : memref<16xi32>)
        aie.use_lock(%lock_0_2_2, Release, 1)
      }]
      aie.end
    }

    // the absolutely only thing that's relevant here is (MM2S, 0, 0) and (S2MM, 0, 0)
    memref.global "public" @this_just_creates_a_symbol_and_the_type_means_nothing_in : memref<1xi32>
    memref.global "public" @this_just_creates_a_symbol_and_the_type_means_nothing_out : memref<1xi32>
    aie.shim_dma_allocation @this_just_creates_a_symbol_and_the_type_means_nothing_in(MM2S, 0, 0)
    aie.shim_dma_allocation @this_just_creates_a_symbol_and_the_type_means_nothing_out(S2MM, 0, 0)
    func.func @bobsyouruncle(%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 0 : i64, metadata = @this_just_creates_a_symbol_and_the_type_means_nothing_in} : memref<64xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 1 : i64, metadata = @this_just_creates_a_symbol_and_the_type_means_nothing_out} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}
