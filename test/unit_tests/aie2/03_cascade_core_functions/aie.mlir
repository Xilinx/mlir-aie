//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano
// RUN: clang --target=aie2 -c %S/kernel.cc
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

// CHECK: PASS!

module {
  AIE.device(xcve2802) {
    %tile13 = AIE.tile(1, 3)
    %tile23 = AIE.tile(2, 3)

    %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
    %buf23_0 = AIE.buffer(%tile23) { sym_name = "c" } : memref<256xi32>

    %lock13_3 = AIE.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
    %lock23_7 = AIE.lock(%tile23, 7) { sym_name = "output_lock" } // output buffer lock

    func.func private @do_mul(%A: memref<256xi32>) -> ()
    func.func private @do_mac(%A: memref<256xi32>) -> ()

    %core13 = AIE.core(%tile13) {
      AIE.useLock(%lock13_3, AcquireGreaterEqual, 1) // acquire for read(e.g. input ping)
      func.call @do_mul(%buf13_0) : (memref<256xi32>) -> ()
      AIE.end
    } { link_with="kernel.o" }

    %core23 = AIE.core(%tile23) {
  //    %val1 = arith.constant 7 : i32
  //    %idx1 = arith.constant 0 : index
  //    memref.store %val1, %buf14_0[%idx1] : memref<256xi32>
      func.call @do_mac(%buf23_0) : (memref<256xi32>) -> ()
      AIE.useLock(%lock23_7, Release, 1) // release for read
      AIE.end
    } { link_with="kernel.o" }
  }
}
