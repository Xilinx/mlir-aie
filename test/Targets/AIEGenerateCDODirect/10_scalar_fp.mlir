// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// RUN: export BASENAME=$(basename %s)
// RUN: rm -rf *.elf* *.xclbin *.bin $BASENAME.cdo_direct $BASENAME.prj
// RUN: mkdir $BASENAME.prj && pushd $BASENAME.prj && %python aiecc.py --aie-generate-cdo --no-compile-host --tmpdir $PWD %s && popd

module @test {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %a = aie.buffer(%tile_1_3) {sym_name = "a"} : memref<256xf32>
    %core_1_3 = aie.core(%tile_1_3) {
      %cst = arith.constant 7.000000e+00 : f32
      %c3 = arith.constant 3 : index
      %0 = arith.addf %cst, %cst : f32
      memref.store %0, %a[%c3] : memref<256xf32>
      %cst_0 = arith.constant 8.000000e+00 : f32
      %c5 = arith.constant 5 : index
      memref.store %cst_0, %a[%c5] : memref<256xf32>
      %1 = memref.load %a[%c3] : memref<256xf32>
      %c9 = arith.constant 9 : index
      memref.store %1, %a[%c9] : memref<256xf32>
      aie.end
    }
  }
}

