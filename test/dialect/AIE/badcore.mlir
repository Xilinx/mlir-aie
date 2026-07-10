//===- badcore.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.core' op failed to verify that op exists in a core tile

module @test {
  %t1 = aie.tile(4, 0)
  %core = aie.core(%t1) {
    aie.end
  }
}
