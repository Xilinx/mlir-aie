//===- test_error_dma_multi_state.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: (aie-translate --aie-generate-xaie %s 2>&1 || true) | FileCheck %s
// CHECK: acquires/releases the lock in a DMA block from/to multiple states.

module @test_error_dma_multi_state {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %l33_0 = AIE.lock(%t33, 0)
  AIE.mem(%t33) {
    AIE.dmaStart(MM2S, 0, ^bb1, ^end)
  ^bb1:
    AIE.useLock(%l33_0, Acquire, 0)
    // This should fail because only one state can be acquired in a DmaBd
    AIE.useLock(%l33_0, Acquire, 1)
    AIE.useLock(%l33_0, Release, 0)
    AIE.useLock(%l33_0, Release, 1)
    AIE.nextBd ^end
  ^end:
    AIE.end
  }
 }
}
