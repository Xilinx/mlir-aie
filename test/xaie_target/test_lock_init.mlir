//===- test_lock_init.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s
// CHECK: XAieTile_LockAcquire(&(ctx->TileInst[3][3]), 0, 1, 1);

module @test_lock_init {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %l33_0 = AIE.lock(%t33, 0)
  // When written in the top-level module, the useLock is treated as
  // an initialization to the lock.  For example, in the operation below,
  // the lock %l33_0 is to be initialized as Acquired 1 when the host function
  // mlir_aie_initialize_locks(ctx) is invoked.
  AIE.useLock(%l33_0, Acquire, 1)
 }
}
