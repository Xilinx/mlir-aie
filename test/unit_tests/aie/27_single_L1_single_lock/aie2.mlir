//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test.elf -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_board ./test.elf

module @test27_simple_shim_dma_single_lock {
  AIE.device(xcve2802) {
    %tile72 = AIE.tile(7, 3)
    %lockCore = AIE.lock(%tile72, 0)  {init = 0 : i32 , sym_name = "coreLock"}
    %dummyLock = AIE.lock(%tile72, 1) { sym_name = "dummyLock"}
    %buf72_0 = AIE.buffer(%tile72) {sym_name = "aieL1" } : memref<16xi32>

    %core72 = AIE.core(%tile72) {
      %c0 = arith.constant 0 : index

      %constant0 = arith.constant 0 : i32
      %constant7 = arith.constant 7 : i32
      %constant13 = arith.constant 13 : i32
      %constant43 = arith.constant 43 : i32
      %constant47 = arith.constant 47 : i32

      AIE.useLock(%lockCore, "Acquire", 0)
      memref.store %constant7, %buf72_0[%c0] : memref<16xi32>
      AIE.useLock(%lockCore, "Release", 1)

      AIE.useLock(%lockCore, "Acquire", 0)
      memref.store %constant13, %buf72_0[%c0] : memref<16xi32>
      AIE.useLock(%lockCore, "Release", 1)

      AIE.useLock(%lockCore, "Acquire", 0)
      memref.store %constant43, %buf72_0[%c0] : memref<16xi32>
      AIE.useLock(%lockCore, "Release", 1)

      AIE.useLock(%lockCore, "Acquire", 0)
      memref.store %constant47, %buf72_0[%c0] : memref<16xi32>
      AIE.useLock(%lockCore, "Release", 1)

      AIE.end
    }
  }
}
