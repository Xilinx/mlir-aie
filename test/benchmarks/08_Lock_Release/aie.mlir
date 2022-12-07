//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// REQUIRES: xaiev1

module @benchmark06_lock_release {
  %tile13 = AIE.tile(1, 3)

  %l13_0 = AIE.lock(%tile13, 0)

AIE.core(%tile13) {
    AIE.useLock(%l13_0, "Release", 0 , 0)
    AIE.end
  }
}