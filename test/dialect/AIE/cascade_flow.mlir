//===- cascade_flow.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s

module @test {
  aie.device(xcve2802) {
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    aie.cascade_flow(%t13, %t23)
  }
}
