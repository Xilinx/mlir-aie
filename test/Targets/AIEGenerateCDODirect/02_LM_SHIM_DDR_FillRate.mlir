// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @benchmark_02_LM2DDR {
  aie.device(ipu) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 3)
    %buf71_0 = aie.buffer(%tile_0_2) {sym_name = "buf71_0"} : memref<7168xi32>
    %buffer = aie.external_buffer {sym_name = "buffer"} : memref<7168xi32>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 1, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock_0_2, Acquire, 0)
      aie.dma_bd(%buf71_0 : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
    %shim_dma_0_0 = aie.shim_dma(%tile_0_0) {
      %lock_0_0 = aie.lock(%tile_0_0, 2)
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_0, Acquire, 1)
      aie.dma_bd(%buffer : memref<7168xi32>, 0, 7168)
      aie.use_lock(%lock_0_0, Release, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<DMA : 1, South : 2>
    }
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<North : 2, South : 2>
    }
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<North : 2, DMA : 0>
    }
  }
}

