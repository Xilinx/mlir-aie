//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: %run_on_board ./test.elf

module @benchmark11_tile_broadcast_vertical {
aie.device(xcvc1902) {
  %t72 = aie.tile(7, 2)
  %t73 = aie.tile(7, 3)
  %t74 = aie.tile(7,4)
  %t75 = aie.tile(7,5)
}
}
 