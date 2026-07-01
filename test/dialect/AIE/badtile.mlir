//===- badtile.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.tile' op attribute 'col' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0

module @test {
  %t1 = aie.tile(-1, -1)
}
