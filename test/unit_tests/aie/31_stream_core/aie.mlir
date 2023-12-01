//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// CHECK: test start.
// CHECK: PASS!

module {
  %tile13 = AIE.tile(1, 3)
  %tile23 = AIE.tile(2, 3)

  %buf13 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf23 = AIE.buffer(%tile23) { sym_name = "c" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
  %lock23_7 = AIE.lock(%tile23, 7) { sym_name = "output_lock" } // output buffer lock

  func.func private @do_mul(%A: memref<256xi32>) -> ()
  func.func private @do_mac(%A: memref<256xi32>) -> ()

  AIE.flow(%tile13, Core : 0, %tile23, Core : 0)

  %core13 = AIE.core(%tile13) {
    %0 = arith.constant 0 : i32
    %idx0 = arith.constant 3 : index
    AIE.useLock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    %val = memref.load %buf13[%idx0] : memref<256xi32>
    AIE.putStream(%0 : i32, %val : i32)
    AIE.useLock(%lock13_3, "Release", 0) // release for write
    AIE.end
  }

  %core23 = AIE.core(%tile23) {
    %0 = arith.constant 0 : i32
    %idx0 = arith.constant 3 : index
    AIE.useLock(%lock23_7, "Acquire", 0) // acquire for write
    %val = AIE.getStream(%0 : i32) : i32
    memref.store %val, %buf23[%idx0] : memref<256xi32>
    AIE.useLock(%lock23_7, "Release", 1) // release for read
    AIE.end
  }

}
