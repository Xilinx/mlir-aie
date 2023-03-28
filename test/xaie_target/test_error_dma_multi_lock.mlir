//===- test_error_dma_multi_lock.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: (aie-translate --aie-generate-xaie %s 2>&1 || true) | FileCheck %s
// CHECK: used in a DMA block that have multiple locks.

module @test_error_dma_multi_lock {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %l33_0 = AIE.lock(%t33, 0)
  %l33_1 = AIE.lock(%t33, 1)
  AIE.mem(%t33) {
    AIE.dmaStart(MM2S, 0, ^bb1, ^end)
  ^bb1:
    AIE.useLock(%l33_0, Acquire, 1)
    // This should fail because only one lock can be used in a DmaBd
    AIE.useLock(%l33_1, Acquire, 1)
    AIE.useLock(%l33_0, Release, 0)
    AIE.useLock(%l33_1, Release, 0)
    AIE.nextBd ^end
  ^end:
    AIE.end
  }
 }
}
