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

module @benchmark09_shim_broadcast {

  %t70 = aie.tile(7, 0)
  %t72 = aie.tile(7, 2)
  %t3 = aie.tile(3,0)
  %60 = aie.tile(6,0)
  aie.lock(%t72, 1) { sym_name = "lock1" }
  aie.lock(%t72, 2) { sym_name = "lock2" }
  
}
