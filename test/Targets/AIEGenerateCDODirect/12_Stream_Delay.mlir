// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $BASENAME.prj %s
// RUN: mkdir $BASENAME.cdo_direct && cp $BASENAME.prj/*.elf $BASENAME.cdo_direct
// RUN: aie-translate --aie-generate-cdo-direct $BASENAME.prj/input_physical.mlir --work-dir-path=$BASENAME.cdo_direct
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_elfs.bin $BASENAME.prj/aie_cdo_elfs.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_enable.bin $BASENAME.prj/aie_cdo_enable.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_error_handling.bin $BASENAME.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_init.bin $BASENAME.prj/aie_cdo_init.bin

module @test12_stream_delay {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %a13 = aie.buffer(%tile_1_3) {sym_name = "a13"} : memref<512xi32>
    %input_lock = aie.lock(%tile_1_3, 5) {sym_name = "input_lock"}
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<DMA : 0, East : 1>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<West : 1, East : 1>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<West : 1, East : 1>
    }
    %switchbox_4_3 = aie.switchbox(%tile_4_3) {
      aie.connect<West : 1, DMA : 1>
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
    %lock_4_3 = aie.lock(%tile_4_3, 6)
    %lock_4_3_0 = aie.lock(%tile_4_3, 7)
    %a43 = aie.buffer(%tile_4_3) {sym_name = "a43"} : memref<512xi32>
    %b43 = aie.buffer(%tile_4_3) {sym_name = "b43"} : memref<256xi32>
    %mem_4_3 = aie.mem(%tile_4_3) {
      %0 = aie.dma_start(S2MM, 1, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock_4_3, Acquire, 0)
      aie.dma_bd(%a43 : memref<512xi32>, 0, 512)
      aie.use_lock(%lock_4_3, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
  }
}

