//===- badlock2.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.lock' op lock assigned invalid id (maximum is 63)
module @test {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %lock = aie.lock(%t1, 64) { sym_name = "lock1" }
  }
}
