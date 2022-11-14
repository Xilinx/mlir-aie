//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf
// REQUIRES: xaiev1

module @benchmark11_tile_broadcast_vertical {
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7,4)
  %t75 = AIE.tile(7,5)
  

}
 