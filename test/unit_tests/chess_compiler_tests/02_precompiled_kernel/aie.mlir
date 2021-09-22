//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%S/../../../../runtime_lib/ %S/../../../../runtime_lib/test_library.cpp %S/test.cpp -o test.elf
// RUN: xchesscc -p me -P ${CARDANO}/data/cervino/lib +l acdc_project/core_1_3.bcf %S/kernel.cc -o custom_1_3.elf
// RUN: %run_on_board ./test.elf

module @test_chess_02_deprecated_precompiled_kernel {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3)
  %lock13_5 = AIE.lock(%tile13, 5)

  %core13 = AIE.core(%tile13) { AIE.end } { elf_file = "custom_1_3.elf" }

}
