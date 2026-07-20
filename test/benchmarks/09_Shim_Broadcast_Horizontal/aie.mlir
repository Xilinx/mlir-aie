//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @benchmark09_shim_broadcast {
aie.device(xcvc1902) {

  %t70 = aie.tile(7, 0)
  %t72 = aie.tile(7, 2)
  %t3 = aie.tile(3,0)
  %60 = aie.tile(6,0)
  aie.lock(%t72, 1) { sym_name = "lock1" }
  aie.lock(%t72, 2) { sym_name = "lock2" }
  
}
}
