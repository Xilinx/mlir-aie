//===- badtile.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not %PYTHON aiecc.py --basic-alloc-scheme %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.tile' op attribute 'col' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0

module @test {
  %t1 = aie.tile(-1, -1)
}
