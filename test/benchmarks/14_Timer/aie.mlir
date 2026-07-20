//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @benchmark14_timer {
aie.device(xcvc1902) {

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
}
