// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename -s .dma_op.mlir %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.dma_op.prj $BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_start.prj && pushd $BASENAME.dma_start.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %S/$BASENAME.dma_start && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_start.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_start.prj

// RUN: mkdir $BASENAME.dma_op.prj && pushd $BASENAME.dma_op.prj && %python aiecc.py --no-compile-host --tmpdir $PWD %s && popd
// RUN: aie-translate --aie-generate-cdo $BASENAME.dma_op.prj/input_physical.mlir --work-dir-path=$BASENAME.dma_op.prj

// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_error_handling.bin $BASENAME.dma_start.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.dma_op.prj/aie_cdo_init.bin $BASENAME.dma_start.prj/aie_cdo_init.bin

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
      aie.dma(MM2S, 0) {loop = false} [{
        aie.use_lock(%input_lock, Acquire, 1)
        aie.dma_bd(%a13 : memref<512xi32>, 0, 512)
        aie.use_lock(%input_lock, Release, 0)
      }]
      aie.end
    }
    %lock_1_4 = aie.lock(%tile_1_4, 6)
    %lock_1_4_0 = aie.lock(%tile_1_4, 7)
    %a14 = aie.buffer(%tile_1_4) {sym_name = "a14"} : memref<512xi32>
    %b14 = aie.buffer(%tile_1_4) {sym_name = "b14"} : memref<256xi32>
    %mem_1_4 = aie.mem(%tile_1_4) {
      aie.dma(S2MM, 1) {loop = false} [{
        aie.use_lock(%lock_1_4, Acquire, 0)
        aie.dma_bd(%a14 : memref<512xi32>, 0, 512)
        aie.use_lock(%lock_1_4, Release, 1)
      }]
      aie.end
    }
  }
}

