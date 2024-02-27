// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin

module @test04_tile_tiledma {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %a13 = aie.buffer(%tile_1_3) {sym_name = "a13"} : memref<512xi32>
    %input_lock = aie.lock(%tile_1_3, 5) {sym_name = "input_lock"}
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<DMA : 0, North : 1>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<South : 1, DMA : 1>
    }
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%input_lock, Acquire, 1)
      aie.dma_bd(%a13 : memref<512xi32>, 0, 512)
      aie.use_lock(%input_lock, Release, 0)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
    %lock_1_4 = aie.lock(%tile_1_4, 6)
    %lock_1_4_0 = aie.lock(%tile_1_4, 7)
    %a14 = aie.buffer(%tile_1_4) {sym_name = "a14"} : memref<512xi32>
    %b14 = aie.buffer(%tile_1_4) {sym_name = "b14"} : memref<256xi32>
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(S2MM, 1, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock_1_4, Acquire, 0)
      aie.dma_bd(%a14 : memref<512xi32>, 0, 512)
      aie.use_lock(%lock_1_4, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
  }
}

