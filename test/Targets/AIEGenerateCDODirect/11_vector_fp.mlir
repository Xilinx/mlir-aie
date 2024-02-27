// (c) Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: aie-translate --aie-generate-cdo %s --cdo-axi-debug 2>&1 | FileCheck %s

// CHECK: Generating: {{.*}}aie_cdo_error_handling.bin
// CHECK: Generating: {{.*}}aie_cdo_init.bin

module @test {
  aie.device(ipu) {
    %tile_1_3 = aie.tile(1, 3)
    %a = aie.buffer(%tile_1_3) {sym_name = "a"} : memref<256xf32>
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
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
      scf.for %arg0 = %c0 to %c64 step %c8 {
        %cst_1 = arith.constant 0.000000e+00 : f32
        %2 = vector.transfer_read %a[%arg0], %cst_1 : memref<256xf32>, vector<8xf32>
        %3 = vector.transfer_read %a[%arg0], %cst_1 : memref<256xf32>, vector<8xf32>
        %4 = arith.mulf %2, %3 : vector<8xf32>
        vector.transfer_write %4, %a[%arg0] : vector<8xf32>, memref<256xf32>
      }
      aie.end
    }
  }
}

