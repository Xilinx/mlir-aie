//===- badlock2.mlir -------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not %PYTHON aiecc.py %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'AIE.lock' op lock assigned invalid id (maximum is 15)
module @test {
  %t1 = AIE.tile(1, 1)
  %lock = AIE.lock(%t1, 16) { sym_name = "lock1" }
}
