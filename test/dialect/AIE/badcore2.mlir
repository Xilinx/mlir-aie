//===- badcore2.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error: 'AIE.core' op CoreOp cannot be created on mem tile

module @test {
  AIE.device(xcve2802) {
    %t1 = AIE.tile(4, 1)
    %core = AIE.core(%t1) {
      AIE.end
    }
  }
}
