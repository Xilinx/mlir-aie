//===- aie_row.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, !hsa
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// RUN: aie_row.mlir.prj/aiesim.sh | FileCheck %s

// XFAIL: *

// CHECK: test start.
// CHECK: PASS!

module @test4_row_shared_memory {
  %tile23 = aie.tile(2, 3)
  %tile33 = aie.tile(3, 3)

  %buf13_0 = aie.buffer(%tile23) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = aie.buffer(%tile23) { sym_name = "b" } : memref<256xi32>
  %buf33_0 = aie.buffer(%tile33) { sym_name = "c" } : memref<256xi32>

  %lock23_3 = aie.lock(%tile23, 3) { sym_name = "input_lock" } // input buffer lock
  %lock23_5 = aie.lock(%tile23, 5) { sym_name = "inter_lock" } // interbuffer lock
  %lock33_7 = aie.lock(%tile33, 7) { sym_name = "output_lock" } // output buffer lock

  %core13 = aie.core(%tile23) {
    aie.use_lock(%lock23_3, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock23_5, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 3 : index
    %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf13_1[%idx2] : memref<256xi32>
    aie.use_lock(%lock23_3, "Release", 0) // release for write
    aie.use_lock(%lock23_5, "Release", 1) // release for read
    aie.end
  }


  %core33 = aie.core(%tile33) {
  aie.use_lock(%lock23_5, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock33_7, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 3 : index
    %val1 = memref.load %buf13_1[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf33_0[%idx2] : memref<256xi32>
    aie.use_lock(%lock23_5, "Release", 0) // release for write
    aie.use_lock(%lock33_7, "Release", 1) // release for read
    aie.end
  }

}
