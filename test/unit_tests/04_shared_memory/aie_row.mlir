//===- aie_row.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --tmpdir=%t --sysroot=%VITIS_SYSROOT% %s -I%S/../../../runtime_lib/ %S/../../../runtime_lib/test_library.cpp %S/test.cpp -o test.elf

module @test4_row_shared_memory {
  %tile13 = AIE.tile(1, 3)
  %tile23 = AIE.tile(2, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xi32>
  %buf23_0 = AIE.buffer(%tile23) { sym_name = "c" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) // input buffer lock
  %lock13_5 = AIE.lock(%tile13, 5) // interbuffer lock
  %lock23_7 = AIE.lock(%tile23, 7) // output buffer lock

  %core13 = AIE.core(%tile13) {
    AIE.useLock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    AIE.useLock(%lock13_5, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 3 : index
    %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf13_1[%idx2] : memref<256xi32>
    AIE.useLock(%lock13_3, "Release", 0) // release for write
    AIE.useLock(%lock13_5, "Release", 1) // release for read
    AIE.end
  }


  %core23 = AIE.core(%tile23) {
    AIE.useLock(%lock13_5, "Acquire", 1) // acquire for read(e.g. input ping)
    AIE.useLock(%lock23_7, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 3 : index
    %val1 = memref.load %buf13_1[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf23_0[%idx2] : memref<256xi32>
    AIE.useLock(%lock13_5, "Release", 0) // release for write
    AIE.useLock(%lock23_7, "Release", 1) // release for read
    AIE.end
  }

}
