//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license, !hsa
// RUN: xchesscc_wrapper aie -c %S/kernel.cc
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: test start.
// CHECK: PASS!

module {
  %tile13 = aie.tile(1, 3)
  %tile23 = aie.tile(2, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf23_0 = aie.buffer(%tile23) { sym_name = "c" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" } // input buffer lock
  %lock23_7 = aie.lock(%tile23, 7) { sym_name = "output_lock" } // output buffer lock
  
  func.func private @do_mul(%A: memref<256xi32>) -> ()
  func.func private @do_mac(%A: memref<256xi32>) -> ()
  
  %core13 = aie.core(%tile13) {
    aie.use_lock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    func.call @do_mul(%buf13_0) : (memref<256xi32>) -> ()
    aie.use_lock(%lock13_3, "Release", 0) // release for write
    aie.end
  } { link_with="kernel.o" }
  
  %core23 = aie.core(%tile23) {
//    %val1 = arith.constant 7 : i32
//    %idx1 = arith.constant 0 : index
//    memref.store %val1, %buf14_0[%idx1] : memref<256xi32>
     aie.use_lock(%lock23_7, "Acquire", 0) // acquire for write
    func.call @do_mac(%buf23_0) : (memref<256xi32>) -> ()
     aie.use_lock(%lock23_7, "Release", 1) // release for read
    aie.end
  } { link_with="kernel.o" }
  
}
