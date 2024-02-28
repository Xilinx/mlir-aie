// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// REQUIRES: cdo_direct_generation
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @test22_init_locks {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %a = aie.buffer(%tile_1_3) {sym_name = "a"} : memref<256xi32>
    %b = aie.buffer(%tile_1_3) {sym_name = "b"} : memref<256xi32>
    %lock_a = aie.lock(%tile_1_3, 3) {init = 1 : i32, sym_name = "lock_a"}
    %lock_b = aie.lock(%tile_1_3, 5) {sym_name = "lock_b"}
    %core_1_3 = aie.core(%tile_1_3) {
      aie.use_lock(%lock_a, Acquire, 1)
      aie.use_lock(%lock_b, Acquire, 0)
      %c3 = arith.constant 3 : index
      %0 = memref.load %a[%c3] : memref<256xi32>
      %1 = arith.addi %0, %0 : i32
      %2 = arith.addi %1, %0 : i32
      %3 = arith.addi %2, %0 : i32
      %4 = arith.addi %3, %0 : i32
      %c5 = arith.constant 5 : index
      memref.store %4, %b[%c5] : memref<256xi32>
      aie.use_lock(%lock_a, Release, 0)
      aie.use_lock(%lock_b, Release, 1)
      aie.end
    }
  }
}

