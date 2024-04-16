//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% %S/test.cpp -o test.elf -L%host_runtime_lib%/test_lib/lib -ltest_lib
// RUN: %run_on_vck5000 ./test.elf

module @test03_sync_with_locks {
  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf13_1 = aie.buffer(%tile13) { sym_name = "b" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "lock1" }
  %lock13_5 = aie.lock(%tile13, 5) { sym_name = "lock2" }

  %core13 = aie.core(%tile13) {
    aie.use_lock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    aie.use_lock(%lock13_5, "Acquire", 0) // acquire for write
    %idx1 = arith.constant 3 : index
    %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    memref.store %5, %buf13_1[%idx2] : memref<256xi32>
    aie.use_lock(%lock13_3, "Release", 0) // release for write
    aie.use_lock(%lock13_5, "Release", 1) // release for read
    aie.end
  }
}
