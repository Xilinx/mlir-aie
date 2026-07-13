//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark07_lock_acquire {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  %l13_0 = aie.lock(%tile13, 0)

aie.core(%tile13) {
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(%l13_0, "Acquire", %c0_ul0)
    aie.end
  }

}
}