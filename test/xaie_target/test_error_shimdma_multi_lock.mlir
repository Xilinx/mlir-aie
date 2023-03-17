//===- test_error_shimdma_multi_lock.mlir ----------------------*- MLIR -*-===//
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

module @test_error_shimdma_multi_lock {
 AIE.device(xcvc1902) {
  %t30 = AIE.tile(3, 0)
  %l30_0 = AIE.lock(%t30, 0)
  %l30_1 = AIE.lock(%t30, 1)
  AIE.shimDMA(%t30) {
    AIE.dmaStart(MM2S, 0, ^bb1, ^end)
  ^bb1:
    AIE.useLock(%l30_0, Acquire, 1)
    // This should fail because only one state can be acquired in a ShimBd
    AIE.useLock(%l30_1, Acquire, 1)
    AIE.useLock(%l30_0, Release, 0)
    AIE.useLock(%l30_1, Release, 0)
    AIE.nextBd ^end
  ^end:
    AIE.end
  }
 }
}
