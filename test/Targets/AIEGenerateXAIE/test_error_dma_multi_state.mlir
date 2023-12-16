//===- test_error_dma_multi_state.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: (aie-translate --aie-generate-xaie %s 2>&1 || true) | FileCheck %s
// CHECK: acquires/releases the lock in a DMA block from/to multiple states.

module @test_error_dma_multi_state {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %l33_0 = AIE.lock(%t33, 0)
  AIE.mem(%t33) {
    AIE.dma_start(MM2S, 0, ^bb1, ^end)
  ^bb1:
    AIE.use_lock(%l33_0, Acquire, 0)
    // This should fail because only one state can be acquired in a DmaBd
    AIE.use_lock(%l33_0, Acquire, 1)
    AIE.use_lock(%l33_0, Release, 0)
    AIE.use_lock(%l33_0, Release, 1)
    AIE.next_bd ^end
  ^end:
    AIE.end
  }
 }
}
