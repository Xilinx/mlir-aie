// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @test04_shared_memory {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %a = aie.buffer(%tile_1_3) {sym_name = "a"} : memref<256xi32>
    %b = aie.buffer(%tile_1_3) {sym_name = "b"} : memref<256xi32>
    %c = aie.buffer(%tile_1_4) {sym_name = "c"} : memref<256xi32>
    %test_lock = aie.lock(%tile_1_3, 2) {sym_name = "test_lock"}
    %input_write_lock = aie.lock(%tile_1_3, 3) {init = 1 : i32, sym_name = "input_write_lock"}
    %input_read_lock = aie.lock(%tile_1_3, 4) {sym_name = "input_read_lock"}
    %hidden_write_lock = aie.lock(%tile_1_3, 5) {init = 1 : i32, sym_name = "hidden_write_lock"}
    %hidden_read_lock = aie.lock(%tile_1_3, 6) {sym_name = "hidden_read_lock"}
    %output_write_lock = aie.lock(%tile_1_4, 7) {init = 1 : i32, sym_name = "output_write_lock"}
    %output_read_lock = aie.lock(%tile_1_4, 8) {sym_name = "output_read_lock"}
    %core_1_3 = aie.core(%tile_1_3) {
      aie.use_lock(%input_read_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%hidden_write_lock, AcquireGreaterEqual, 1)
      %c3 = arith.constant 3 : index
      %0 = memref.load %a[%c3] : memref<256xi32>
      %1 = arith.addi %0, %0 : i32
      %2 = arith.addi %1, %0 : i32
      %3 = arith.addi %2, %0 : i32
      %4 = arith.addi %3, %0 : i32
      %c5 = arith.constant 5 : index
      memref.store %4, %b[%c5] : memref<256xi32>
      aie.use_lock(%input_write_lock, Release, 1)
      aie.use_lock(%hidden_read_lock, Release, 1)
      aie.end
    }
    %core_1_4 = aie.core(%tile_1_4) {
      aie.use_lock(%hidden_read_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%output_write_lock, AcquireGreaterEqual, 1)
      %c5 = arith.constant 5 : index
      %0 = memref.load %b[%c5] : memref<256xi32>
      %1 = arith.addi %0, %0 : i32
      %2 = arith.addi %1, %0 : i32
      %3 = arith.addi %2, %0 : i32
      %4 = arith.addi %3, %0 : i32
      %c5_0 = arith.constant 5 : index
      memref.store %4, %c[%c5_0] : memref<256xi32>
      aie.use_lock(%hidden_write_lock, Release, 1)
      aie.use_lock(%output_read_lock, Release, 1)
      aie.end
    }
  }
}

