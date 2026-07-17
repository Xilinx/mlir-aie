//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @benchmark05_core_startup {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>

  %core13 = aie.core(%tile13) {
    aie.end
  }
}
}
