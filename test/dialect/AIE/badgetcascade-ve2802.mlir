//===- badcore.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.get_cascade' op must be a 512-bit type

module @test {
  aie.device(xcve2802) {
    %t33 = aie.tile(3, 3)
    %c33 = aie.core(%t33) {
      %val2 = aie.get_cascade() : i64
      aie.end
    }
  }
}
