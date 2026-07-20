//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test02_lock_acquire_release {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>

  %lock13_3 = aie.lock(%tile13, 3) { sym_name = "lock1" }
  %lock13_5 = aie.lock(%tile13, 5) { sym_name = "lock2" }

  %core13 = aie.core(%tile13) {
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(%lock13_3, "Acquire", %c0_ul0) // acquire for write (e.g. input ping)
    %c0_ul1 = arith.constant 0 : i32
    aie.use_lock(%lock13_5, "Acquire", %c0_ul1) // acquire for write (e.g. input ping)
    %c1_ul2 = arith.constant 1 : i32
    aie.use_lock(%lock13_5, "Release", %c1_ul2) // release for read
    aie.end
  }

}
}
