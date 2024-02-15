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

module @test_chess_08_tile_locks {
  aie.device(ipu) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %west = aie.buffer(%tile_0_1) {sym_name = "west"} : memref<256xi32>
    %local = aie.buffer(%tile_1_1) {sym_name = "local"} : memref<256xi32>
    %east = aie.buffer(%tile_2_1) {sym_name = "east"} : memref<256xi32>
    %start_lock_1 = aie.lock(%tile_1_1, 0) {sym_name = "start_lock_1"}
    %done_lock_1 = aie.lock(%tile_1_1, 1) {sym_name = "done_lock_1"}
    %start_lock_2 = aie.lock(%tile_1_1, 2) {sym_name = "start_lock_2"}
    %done_lock_2 = aie.lock(%tile_1_1, 3) {sym_name = "done_lock_2"}
    aie.flow(%tile_1_1, DMA : 0, %tile_1_1, DMA : 0)
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      aie.dma(MM2S, 0) {loop = false} [{
        aie.use_lock(%done_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%east : memref<256xi32>, 0, 2)
        aie.use_lock(%start_lock_2, Release, 1)
      }]
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      aie.dma(MM2S, 0) {loop = false} [{
        aie.use_lock(%start_lock_1, AcquireGreaterEqual, 1)
        aie.dma_bd(%west : memref<256xi32>, 0, 2)
        aie.use_lock(%done_lock_1, Release, 1)
      }, {
        aie.use_lock(%start_lock_1, AcquireGreaterEqual, 1)
        aie.dma_bd(%west : memref<256xi32>, 4, 2)
        aie.use_lock(%done_lock_1, Release, 1)
      }]
      aie.dma(S2MM, 0) {loop = false} [{
        aie.use_lock(%start_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%east : memref<256xi32>, 8, 2)
        aie.use_lock(%done_lock_2, Release, 1)
      }, {
        aie.use_lock(%start_lock_2, AcquireGreaterEqual, 1)
        aie.dma_bd(%east : memref<256xi32>, 12, 2)
        aie.use_lock(%done_lock_2, Release, 1)
      }]
      aie.end
    }
  }
}

