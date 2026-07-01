//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark10_tile_broadcast_horizontal {
aie.device(xcvc1902) {
  %t63 = aie.tile(6, 3)
  %t73 = aie.tile(7, 3)
  %t83 = aie.tile(8,3)
}
}
