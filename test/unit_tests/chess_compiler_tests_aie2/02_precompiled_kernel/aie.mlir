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
// RUN: %PYTHON aiecc.py --aiesim --xchesscc --xbridge --no-compile-host %s %test_lib_flags %S/test.cpp
// RUN: xchesscc_wrapper aie2 +l aie.mlir.prj/core_1_3.bcf %S/kernel.cc -o custom_1_3.elf
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2 ISS
// CHECK: test start.
// CHECK: PASS!

module @test_chess_02_deprecated_precompiled_kernel {
  aie.device(xcve2802) {
    %tile13 = aie.tile(1, 3)

    %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
    %buf13_1 = aie.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

    %lock13_3 = aie.lock(%tile13, 3) { sym_name = "input_lock" }
    %lock13_5 = aie.lock(%tile13, 5) { sym_name = "output_lock" }

    %core13 = aie.core(%tile13) { aie.end } { elf_file = "custom_1_3.elf" }
  }
}
