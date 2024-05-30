//===- test_error_dma_multi_state.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: (aie-translate --aie-generate-xaie %s 2>&1 || true) | FileCheck %s
// CHECK: acquires/releases the lock in a DMA block from/to multiple states.

module @test_error_dma_multi_state {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %l33_0 = aie.lock(%t33, 0)
  aie.mem(%t33) {
    aie.dma_start(MM2S, 0, ^bb1, ^end)
  ^bb1:
    aie.use_lock(%l33_0, Acquire, 0)
    // This should fail because only one state can be acquired in a DmaBd
    aie.use_lock(%l33_0, Acquire, 1)
    aie.use_lock(%l33_0, Release, 0)
    aie.use_lock(%l33_0, Release, 1)
    aie.next_bd ^end
  ^end:
    aie.end
  }
 }
}
