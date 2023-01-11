//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py -j4 --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib%/ %aie_runtime_lib%/test_library.cpp %S/test.cpp -o tutorial-7.exe
// RUN: %run_on_board ./tutorial-7.exe
