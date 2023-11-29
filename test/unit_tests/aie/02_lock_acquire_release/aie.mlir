//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test.elf -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_board ./test.elf

module @test02_lock_acquire_release {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) { sym_name = "lock1" }
  %lock13_5 = AIE.lock(%tile13, 5) { sym_name = "lock2" }

  %core13 = AIE.core(%tile13) {
    AIE.useLock(%lock13_3, "Acquire", 0) // acquire for write (e.g. input ping)
    AIE.useLock(%lock13_5, "Acquire", 0) // acquire for write (e.g. input ping)
    AIE.useLock(%lock13_5, "Release", 1) // release for read
    AIE.end
  }

}
