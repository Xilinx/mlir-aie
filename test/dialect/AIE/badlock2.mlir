//===- badlock2.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error: 'AIE.lock' op attribute 'lockID' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0
module @test {
  %t1 = AIE.tile(1, 1)
  %lock = AIE.lock(%t1, -3) { sym_name = "lock1" }

  AIE.core(%t1) {
    AIE.useLock(%lock, "Acquire", 1)
    AIE.end
  }
}
