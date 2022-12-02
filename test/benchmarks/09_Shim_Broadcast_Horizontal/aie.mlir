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

module @benchmark09_shim_broadcast {

  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)
  %t3 = AIE.tile(3,0)
  %60 = AIE.tile(6,0)
  AIE.lock(%t72, 1) { sym_name = "lock1" }
  AIE.lock(%t72, 2) { sym_name = "lock2" }
  
}
