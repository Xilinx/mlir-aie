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

module {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %a = aie.buffer(%tile_1_3) {sym_name = "a"} : memref<256xi32>
    %c = aie.buffer(%tile_2_3) {sym_name = "c"} : memref<256xi32>
    %input_lock = aie.lock(%tile_1_3, 3) {sym_name = "input_lock"}
    %output_lock = aie.lock(%tile_2_3, 7) {sym_name = "output_lock"}
    func.func private @do_mul(memref<256xi32>)
    func.func private @do_mac(memref<256xi32>)
    aie.flow(%tile_1_3, Core : 0, %tile_2_3, Core : 0)
    %core_1_3 = aie.core(%tile_1_3) {
      %c0_i32 = arith.constant 0 : i32
      %c3 = arith.constant 3 : index
      aie.use_lock(%input_lock, Acquire, 1)
      %0 = memref.load %a[%c3] : memref<256xi32>
      aie.put_stream(%c0_i32 : i32, %0 : i32)
      aie.use_lock(%input_lock, Release, 0)
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c0_i32 = arith.constant 0 : i32
      %c3 = arith.constant 3 : index
      aie.use_lock(%output_lock, Acquire, 0)
      %0 = aie.get_stream(%c0_i32 : i32) : i32
      memref.store %0, %c[%c3] : memref<256xi32>
      aie.use_lock(%output_lock, Release, 1)
      aie.end
    }
  }
}

