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

module @test27_simple_shim_dma_single_lock {
  aie.device(ipu) {
    %tile_0_3 = aie.tile(0, 3)
    %coreLock = aie.lock(%tile_0_3, 0) {sym_name = "coreLock"}
    %dummyLock = aie.lock(%tile_0_3, 1) {sym_name = "dummyLock"}
    %aieL1 = aie.buffer(%tile_0_3) {sym_name = "aieL1"} : memref<16xi32>
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c7_i32 = arith.constant 7 : i32
      %c13_i32 = arith.constant 13 : i32
      %c43_i32 = arith.constant 43 : i32
      %c47_i32 = arith.constant 47 : i32
      aie.use_lock(%coreLock, Acquire, 0)
      memref.store %c7_i32, %aieL1[%c0] : memref<16xi32>
      aie.use_lock(%coreLock, Release, 1)
      aie.use_lock(%coreLock, Acquire, 0)
      aie.use_lock(%coreLock, Release, 1)
      aie.use_lock(%coreLock, Acquire, 0)
      memref.store %c13_i32, %aieL1[%c0] : memref<16xi32>
      aie.use_lock(%coreLock, Release, 1)
      aie.use_lock(%coreLock, Acquire, 0)
      aie.use_lock(%coreLock, Release, 1)
      aie.use_lock(%coreLock, Acquire, 0)
      memref.store %c43_i32, %aieL1[%c0] : memref<16xi32>
      aie.use_lock(%coreLock, Release, 1)
      aie.use_lock(%coreLock, Acquire, 0)
      aie.use_lock(%coreLock, Release, 1)
      aie.use_lock(%coreLock, Acquire, 0)
      memref.store %c47_i32, %aieL1[%c0] : memref<16xi32>
      aie.use_lock(%coreLock, Release, 1)
      aie.use_lock(%coreLock, Acquire, 0)
      aie.use_lock(%coreLock, Release, 1)
      aie.end
    }
  }
}

