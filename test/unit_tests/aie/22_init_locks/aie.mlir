//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test22_init_locks {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  aie.buffer @a(%tile13) : memref<256xi32>
  aie.buffer @b(%tile13) : memref<256xi32>

  aie.lock @lock_a(%tile13, 3) { init = 1 : i32}
  aie.lock @lock_b(%tile13, 5)

  %core13 = aie.core(%tile13) {
    %c1_ul0 = arith.constant 1 : i32
    aie.use_lock(@lock_a, "Acquire", %c1_ul0) // acquire for read(e.g. input ping)
    %c0_ul1 = arith.constant 0 : i32
    aie.use_lock(@lock_b, "Acquire", %c0_ul1) // acquire for write
    %idx1 = arith.constant 3 : index
    %buf13_0 = aie.get_buffer @a : memref<256xi32>
    %val1 = memref.load %buf13_0[%idx1] : memref<256xi32>
    %2    = arith.addi %val1, %val1 : i32
    %3 = arith.addi %2, %val1 : i32
    %4 = arith.addi %3, %val1 : i32
    %5 = arith.addi %4, %val1 : i32
    %idx2 = arith.constant 5 : index
    %buf13_1 = aie.get_buffer @b : memref<256xi32>
    memref.store %5, %buf13_1[%idx2] : memref<256xi32>
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(@lock_a, "Release", %c0_ul2) // release for write
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(@lock_b, "Release", %c1_ul3) // release for read
    aie.end
  }
}
}
