//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xf32>

  %core13 = AIE.core(%tile13) {
    %c0 = constant 0 : index
    %c64 = constant 64 : index
    %c8 = constant 8 : index
    %val1 = constant 7.0 : f32
    %idx1 = constant 3 : index
    %2 = addf %val1, %val1 : f32
    memref.store %2, %buf13_0[%idx1] : memref<256xf32>
    %val2 = constant 8.0 : f32
    %idx2 = constant 5 : index
    memref.store %val2, %buf13_0[%idx2] : memref<256xf32>
    %val3 = memref.load %buf13_0[%idx1] : memref<256xf32>
    %idx3 = constant 9 : index
    memref.store %val3,%buf13_0[%idx3] : memref<256xf32>
    scf.for %arg0 = %c0 to %c64 step %c8 {
      %cst = constant 0.000000e+00 : f32
      %59 = vector.transfer_read %buf13_0[%arg0], %cst : memref<256xf32>, vector<8xf32>
      %60 = vector.transfer_read %buf13_0[%arg0], %cst : memref<256xf32>, vector<8xf32>
      %61 = mulf %59, %60 : vector<8xf32>
      vector.transfer_write %61, %buf13_0[%arg0] : vector<8xf32>, memref<256xf32>
    }
    AIE.end
  }
}
