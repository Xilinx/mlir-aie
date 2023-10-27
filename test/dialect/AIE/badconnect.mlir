//===- badconnect.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not %PYTHON aiecc.py %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'AIE.connect' op source index cannot be less than zero

module {
  %20 = AIE.tile(2, 0)
  AIE.switchbox(%20) {
    AIE.connect<East: -1, East: 0>
  }
}
