// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd
// RUN: mkdir $BASENAME.cdo_direct && cp $BASENAME.prj/*.elf $BASENAME.cdo_direct
// RUN: aie-translate --aie-generate-cdo-direct $BASENAME.prj/input_physical.mlir --work-dir-path=$BASENAME.cdo_direct
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_elfs.bin $BASENAME.prj/aie_cdo_elfs.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_enable.bin $BASENAME.prj/aie_cdo_enable.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_error_handling.bin $BASENAME.prj/aie_cdo_error_handling.bin
// RUN: cmp $BASENAME.cdo_direct/aie_cdo_init.bin $BASENAME.prj/aie_cdo_init.bin

module @tutorial_2b {
  aie.device(ipu) {
    %tile_1_4 = aie.tile(1, 4)
    %tile_3_4 = aie.tile(3, 4)
    aie.flow(%tile_1_4, DMA : 0, %tile_3_4, DMA : 0)
    %buf14 = aie.buffer(%tile_1_4) {sym_name = "buf14"} : memref<128xi32>
    %buf34 = aie.buffer(%tile_3_4) {sym_name = "buf34"} : memref<128xi32>
    %lock14_done = aie.lock(%tile_1_4, 0) {init = 0 : i32, sym_name = "lock14_done"}
    %lock14_sent = aie.lock(%tile_1_4, 1) {init = 0 : i32, sym_name = "lock14_sent"}
    %lock34_wait = aie.lock(%tile_3_4, 0) {init = 1 : i32, sym_name = "lock34_wait"}
    %lock34_recv = aie.lock(%tile_3_4, 1) {init = 0 : i32, sym_name = "lock34_recv"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = scf.for %arg0 = %c0 to %c128 step %c1 iter_args(%arg1 = %c0_i32) -> (i32) {
        memref.store %arg1, %buf14[%arg0] : memref<128xi32>
        %1 = arith.addi %c1_i32, %arg1 : i32
        scf.yield %1 : i32
      }
      aie.use_lock(%lock14_done, Release, 1)
      aie.end
    }
    %core_3_4 = aie.core(%tile_3_4) {
      aie.end
    }
    %mem_1_4 = aie.mem(%tile_1_4) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock14_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<128xi32>, 0, 128, [<size = 8, stride = 16>, <size = 2, stride = 1>, <size = 8, stride = 2>])
      aie.use_lock(%lock14_sent, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
    %mem_3_4 = aie.mem(%tile_3_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // pred: ^bb0
      aie.use_lock(%lock34_wait, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf34 : memref<128xi32>, 0, 128)
      aie.use_lock(%lock34_recv, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      aie.end
    }
  }
}

