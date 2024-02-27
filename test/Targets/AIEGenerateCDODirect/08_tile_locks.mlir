// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin

module @test_chess_08_tile_locks {
  aie.device(ipu) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_4 = aie.tile(1, 4)
    %east = aie.buffer(%tile_1_3) {sym_name = "east"} : memref<256xi32>
    %north = aie.buffer(%tile_1_3) {sym_name = "north"} : memref<256xi32>
    %south = aie.buffer(%tile_1_3) {sym_name = "south"} : memref<256xi32>
    %local = aie.buffer(%tile_1_3) {sym_name = "local"} : memref<256xi32>
    %start_lock_1 = aie.lock(%tile_1_3, 0) {sym_name = "start_lock_1"}
    %done_lock_1 = aie.lock(%tile_1_3, 1) {sym_name = "done_lock_1"}
    %start_lock_2 = aie.lock(%tile_1_3, 2) {sym_name = "start_lock_2"}
    %done_lock_2 = aie.lock(%tile_1_3, 3) {sym_name = "done_lock_2"}
    aie.flow(%tile_1_3, DMA : 0, %tile_1_3, DMA : 0)
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb2, ^bb1)
    ^bb1:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb6)
    ^bb2:  // pred: ^bb0
      aie.use_lock(%start_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%local : memref<256xi32>, 0, 2)
      aie.use_lock(%done_lock_1, Release, 1)
      aie.next_bd ^bb3
    ^bb3:  // pred: ^bb2
      aie.use_lock(%start_lock_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%local : memref<256xi32>, 4, 2)
      aie.use_lock(%done_lock_1, Release, 1)
      aie.next_bd ^bb6
    ^bb4:  // pred: ^bb1
      aie.use_lock(%start_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%local : memref<256xi32>, 8, 2)
      aie.use_lock(%done_lock_2, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%start_lock_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%local : memref<256xi32>, 12, 2)
      aie.use_lock(%done_lock_2, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // 3 preds: ^bb1, ^bb3, ^bb5
      aie.end
    }
  }
}

