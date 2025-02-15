//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test02_lock_acquire_release {
  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "lock1" }
  %lock13_5 = aie.lock(%tile13, 5) { sym_name = "lock2" }

  %core13 = aie.core(%tile13) {
    aie.use_lock(%lock13_3, "Acquire", 0) // acquire for write (e.g. input ping)
    aie.use_lock(%lock13_5, "Acquire", 0) // acquire for write (e.g. input ping)
    aie.use_lock(%lock13_5, "Release", 1) // release for read
    aie.end
  }

}
