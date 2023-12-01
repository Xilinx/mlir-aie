//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano && jackl
// RUN: clang++ -O2 --target=aie -c -I/usr/include/aie %S/kernel.cpp
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test {
  %tile13 = AIE.tile(1, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xf32>
  %buf13_1 = AIE.buffer(%tile13) { sym_name = "b" } : memref<256xf32>

  %lock13_3 = AIE.lock(%tile13, 3) { sym_name = "inout_lock" }

  func.func private @func(%A: memref<256xf32>, %B: memref<256xf32>) -> ()

  %core13 = AIE.core(%tile13) {
    AIE.useLock(%lock13_3, "Acquire", 1) // acquire
    func.call @func(%buf13_0, %buf13_1) : (memref<256xf32>, memref<256xf32>) -> ()
    AIE.useLock(%lock13_3, "Release", 0) // release for write
    AIE.end
  } { link_with="kernel.o" }
}
