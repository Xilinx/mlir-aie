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
// RUN: %PEANO_INSTALL_DIR/bin/clang++ -O2 --target=aie -c -I/usr/include/aie %S/kernel.cpp
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test {
  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xf32>
  %buf13_1 = aie.buffer(%tile13) { sym_name = "b" } : memref<256xf32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "inout_lock" }

  func.func private @func(%A: memref<256xf32>, %B: memref<256xf32>) -> ()

  %core13 = aie.core(%tile13) {
    aie.use_lock(%lock13_3, "Acquire", 1) // acquire
    func.call @func(%buf13_0, %buf13_1) : (memref<256xf32>, memref<256xf32>) -> ()
    aie.use_lock(%lock13_3, "Release", 0) // release for write
    aie.end
  } { link_with="kernel.o" }
}
