//===- badtile2.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.tile' op column index (50) must be less than the number of columns in the device (38)

module @test {
 aie.device(xcve2802) {
  %t1 = aie.tile(50, 50)
 }
}
