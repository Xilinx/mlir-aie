//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark09_shim_broadcast {

  %t70 = AIE.tile(7, 0)
  %t72 = AIE.tile(7, 2)
  %t3 = AIE.tile(3,0)
  %60 = AIE.tile(6,0)
  AIE.lock(%t72, 1) { sym_name = "lock1" }
  AIE.lock(%t72, 2) { sym_name = "lock2" }
  
}
