//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test {
  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xf32>

  %core13 = aie.core(%tile13) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %val1 = arith.constant 7.0 : f32
    %idx1 = arith.constant 3 : index
    %2 = arith.addf %val1, %val1 : f32
    memref.store %2, %buf13_0[%idx1] : memref<256xf32>
    %val2 = arith.constant 8.0 : f32
    %idx2 = arith.constant 5 : index
    memref.store %val2, %buf13_0[%idx2] : memref<256xf32>
    %val3 = memref.load %buf13_0[%idx1] : memref<256xf32>
    %idx3 = arith.constant 9 : index
    memref.store %val3,%buf13_0[%idx3] : memref<256xf32>
    scf.for %arg0 = %c0 to %c64 step %c8 {
      %cst = arith.constant 0.000000e+00 : f32
      %59 = vector.transfer_read %buf13_0[%arg0], %cst : memref<256xf32>, vector<8xf32>
      %60 = vector.transfer_read %buf13_0[%arg0], %cst : memref<256xf32>, vector<8xf32>
      %61 = arith.mulf %59, %60 : vector<8xf32>
      vector.transfer_write %61, %buf13_0[%arg0] : vector<8xf32>, memref<256xf32>
    }
    aie.end
  }
}
