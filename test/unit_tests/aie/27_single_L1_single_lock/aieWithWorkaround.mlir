//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test27_simple_shim_dma_single_lock {
  %tile73 = AIE.tile(7, 3)
  %lockCore = AIE.lock(%tile73, 0) { sym_name = "coreLock"}
  %dummyLock = AIE.lock(%tile73, 1) { sym_name = "dummyLock"}
  %buf73_0 = AIE.buffer(%tile73) {sym_name = "aieL1" } : memref<16xi32>

  %core72 = AIE.core(%tile73) {
    %c0 = arith.constant 0 : index

    %constant0 = arith.constant 0 : i32
    %constant7 = arith.constant 7 : i32
    %constant13 = arith.constant 13 : i32
    %constant43 = arith.constant 43 : i32
    %constant47 = arith.constant 47 : i32

    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant7, %buf73_0[%c0] : memref<16xi32>
    AIE.useLock(%lockCore, "Release", 1)

    AIE.useLock(%lockCore, "Acquire", 0)
    AIE.useLock(%lockCore, "Release", 1)
    // AIE.useLock(%dummyLock, "Acquire", 0)
    // AIE.useLock(%dummyLock, "Release", 0)

    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant13, %buf73_0[%c0] : memref<16xi32>
    AIE.useLock(%lockCore, "Release", 1)

    AIE.useLock(%lockCore, "Acquire", 0)
    AIE.useLock(%lockCore, "Release", 1)
    // AIE.useLock(%dummyLock, "Acquire", 0)
    // AIE.useLock(%dummyLock, "Release", 0)

    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant43, %buf73_0[%c0] : memref<16xi32>
    AIE.useLock(%lockCore, "Release", 1)

    AIE.useLock(%lockCore, "Acquire", 0)
    AIE.useLock(%lockCore, "Release", 1)
    // AIE.useLock(%dummyLock, "Acquire", 0)
    // AIE.useLock(%dummyLock, "Release", 0)

    AIE.useLock(%lockCore, "Acquire", 0)
    memref.store %constant47, %buf73_0[%c0] : memref<16xi32>
    AIE.useLock(%lockCore, "Release", 1)

    AIE.useLock(%lockCore, "Acquire", 0)
    AIE.useLock(%lockCore, "Release", 1)
    // AIE.useLock(%dummyLock, "Acquire", 0)
    // AIE.useLock(%dummyLock, "Release", 0)

    AIE.end
  }
}
