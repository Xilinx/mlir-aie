// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin
// CHECK: Generating: {{.*}}aie_cdo_init.bin

module @test23_broadcast_packet {
  aie.device(ipu) {
    %tile_1_2 = aie.tile(1, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %buf72_0 = aie.buffer(%tile_1_2) {sym_name = "buf72_0"} : memref<1024xi32>
    %buf72_1 = aie.buffer(%tile_1_2) {sym_name = "buf72_1"} : memref<1024xi32>
    %buf63_0 = aie.buffer(%tile_0_3) {sym_name = "buf63_0"} : memref<1024xi32>
    %buf64_0 = aie.buffer(%tile_0_4) {sym_name = "buf64_0"} : memref<1024xi32>
    %buf73_0 = aie.buffer(%tile_1_3) {sym_name = "buf73_0"} : memref<1024xi32>
    %buf74_0 = aie.buffer(%tile_1_4) {sym_name = "buf74_0"} : memref<1024xi32>
    aiex.broadcast_packet(%tile_1_2, DMA : 0) {
      aiex.bp_id(0) {
        aiex.bp_dest<%tile_1_3, DMA : 0>
        aiex.bp_dest<%tile_0_3, DMA : 0>
      }
      aiex.bp_id(1) {
        aiex.bp_dest<%tile_1_4, DMA : 0>
        aiex.bp_dest<%tile_0_4, DMA : 0>
      }
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %lock_1_2 = aie.lock(%tile_1_2, 4)
      %lock_1_2_0 = aie.lock(%tile_1_2, 5)
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_1_2, Acquire, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buf72_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock_1_2, Release, 0)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_1_2_0, Acquire, 1)
      aie.dma_bd_packet(1, 1)
      aie.dma_bd(%buf72_1 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock_1_2_0, Release, 0)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
    %lock_0_3 = aie.lock(%tile_0_3, 0)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3, Acquire, 0)
      aie.dma_bd(%buf63_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock_0_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %lock_0_4 = aie.lock(%tile_0_4, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4, Acquire, 0)
      aie.dma_bd(%buf64_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock_0_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %lock_1_3 = aie.lock(%tile_1_3, 0)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3, Acquire, 0)
      aie.dma_bd(%buf73_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock_1_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %lock_1_4 = aie.lock(%tile_1_4, 0)
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_4, Acquire, 0)
      aie.dma_bd(%buf74_0 : memref<1024xi32>, 0, 1024)
      aie.use_lock(%lock_1_4, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
  }
}

