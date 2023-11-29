//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test.elf -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_board ./test.elf

// CHECK: PASS!

module @test04_shared_memory {
  AIE.device(xcve2802) {
    %tile13 = AIE.tile(1, 3)
    %tile14 = AIE.tile(1, 4)

    %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
    %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xi32>
    %buf14_0 = AIE.buffer(%tile14) { sym_name = "c" } : memref<256xi32>

    %lock13_3 = AIE.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
    %lock13_5 = AIE.lock(%tile13, 5) { sym_name = "output_lock" } // output buffer lock

    %core13 = AIE.core(%tile13) {
      AIE.useLock(%lock13_3, AcquireGreaterEqual, 1)
      %idx1 = arith.constant 3 : index
      %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
      %2    = arith.addi %val1, %val1 : i32
      %3 = arith.addi %2, %val1 : i32
      %4 = arith.addi %3, %val1 : i32
      %5 = arith.addi %4, %val1 : i32
      %idx2 = arith.constant 5 : index
      memref.store %5, %buf13_1[%idx2] : memref<256xi32>
      AIE.useLock(%lock13_5, Release, 1)
      AIE.end
    }
  }
}
