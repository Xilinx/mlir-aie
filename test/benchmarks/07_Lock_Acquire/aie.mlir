//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2021, Xilinx, Inc. All rights reserved.
// Copyright (C) 2022-2025, Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @benchmark07_lock_acquire {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  %l13_0 = aie.lock(%tile13, 0)

aie.core(%tile13) {
    aie.use_lock(%l13_0, "Acquire", 0)
    aie.end
  }

}
}