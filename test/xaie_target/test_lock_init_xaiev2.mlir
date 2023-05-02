//===- test_lock_init.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie --xaie-target=v2 %s | FileCheck %s
// CHECK: XAie_LockSetValue(&(ctx->DevInst), XAie_TileLoc(3,3), XAie_LockInit(0, 1));

module @test_lock_init {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %l33_0 = AIE.lock(%t33, 0) { init = 1 : i32 }
 }
}
