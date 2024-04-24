//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark13_program_counter {
  %t73 = aie.tile(7, 3)


  %buf73_0 = aie.buffer(%t73) { sym_name = "a" } : memref<256xi32>

  %core73 = aie.core(%t73) {
    %val1 = arith.constant 7 : i32
    %idx1 = arith.constant 3 : index
    %2 = arith.addi %val1, %val1 : i32
    memref.store %2, %buf73_0[%idx1] : memref<256xi32>

    aie.end
  }

}