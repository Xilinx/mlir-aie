//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test02_lock_acquire_release {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  aie.buffer @a(%tile13) : memref<256xi32>

  aie.lock @lock1(%tile13, 3)
  aie.lock @lock2(%tile13, 5)

  %core13 = aie.core(%tile13) {
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(@lock1, "Acquire", %c0_ul0) // acquire for write (e.g. input ping)
    %c0_ul1 = arith.constant 0 : i32
    aie.use_lock(@lock2, "Acquire", %c0_ul1) // acquire for write (e.g. input ping)
    %c1_ul2 = arith.constant 1 : i32
    aie.use_lock(@lock2, "Release", %c1_ul2) // release for read
    aie.end
  }

}
}
