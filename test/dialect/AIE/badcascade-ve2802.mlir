//===- badcore.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'AIE.putCascade' op must be a 512-bit type

module @test {
  AIE.device(xcve2802) {
    %t33 = AIE.tile(3, 3)
    %c33 = AIE.core(%t33) {
      %val2 = arith.constant 1 : i64
      AIE.putCascade(%val2: i64)
      AIE.end
    }
  }
}
