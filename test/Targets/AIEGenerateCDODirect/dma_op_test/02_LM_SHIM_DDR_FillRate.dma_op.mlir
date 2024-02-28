// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename -s .dma_op.mlir %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.dma_op.prj $BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_start.prj && pushd $BASENAME.dma_start.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %S/$BASENAME.dma_start && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_start.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_op.prj && pushd $BASENAME.dma_op.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %s && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_op.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_op.prj

// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_error_handling.bin $BASENAME.dma_start.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_init.bin $BASENAME.dma_start.prj/aie_cdo_init.bin

module @benchmark_02_LM2DDR {
  aie.device(ipu) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %lock_0_2 = aie.lock(%tile_0_2, 3)
    %buf71_0 = aie.buffer(%tile_0_2) {sym_name = "buf71_0"} : memref<7168xi32>
    %buffer = aie.external_buffer {sym_name = "buffer"} : memref<7168xi32>
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma(MM2S, 1) {loop = false} [{
        aie.use_lock(%lock_0_2, Acquire, 0)
        aie.dma_bd(%buf71_0 : memref<7168xi32>, 0, 7168)
        aie.use_lock(%lock_0_2, Release, 1)
      }]
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

