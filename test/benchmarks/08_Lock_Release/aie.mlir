//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark06_lock_release {
  %tile13 = aie.tile(1, 3)

  %l13_0 = aie.lock(%tile13, 0)

aie.core(%tile13) {
    aie.use_lock(%l13_0, "Release", 0)
    aie.end
  }
}