// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin
// CHECK: Generating: {{.*}}aie_cdo_init.bin

module @test18_simple_shim_dma_routed {
  aie.device(ipu) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %input_buffer = aie.external_buffer {sym_name = "input_buffer"} : memref<512xi32>
    %input_lock = aie.lock(%tile_0_0, 1) {sym_name = "input_lock"}
    %shim_dma_0_0 = aie.shim_dma(%tile_0_0) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%input_lock, Acquire, 1)
      aie.dma_bd(%input_buffer : memref<512xi32>, 0, 512)
      aie.use_lock(%input_lock, Release, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    %buf72_0 = aie.buffer(%tile_0_2) {sym_name = "buf72_0"} : memref<256xi32>
    %buf72_1 = aie.buffer(%tile_0_2) {sym_name = "buf72_1"} : memref<256xi32>
    %lock_0_2 = aie.lock(%tile_0_2, 0)
    %lock_0_2_0 = aie.lock(%tile_0_2, 1)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2, Acquire, 0)
      aie.dma_bd(%buf72_0 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_0, Acquire, 0)
      aie.dma_bd(%buf72_1 : memref<256xi32>, 0, 256)
      aie.use_lock(%lock_0_2_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      aie.end
    }
  }
}

