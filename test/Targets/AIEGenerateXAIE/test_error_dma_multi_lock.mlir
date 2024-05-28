//===- test_error_dma_multi_lock.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: (aie-translate --aie-generate-xaie %s 2>&1 || true) | FileCheck %s
// CHECK: used in a DMA block that have multiple locks.

module @test_error_dma_multi_lock {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %l33_0 = aie.lock(%t33, 0)
  %l33_1 = aie.lock(%t33, 1)
  aie.mem(%t33) {
    aie.dma_start(MM2S, 0, ^bb1, ^end)
  ^bb1:
    aie.use_lock(%l33_0, Acquire, 1)
    // This should fail because only one lock can be used in a DmaBd
    aie.use_lock(%l33_1, Acquire, 1)
    aie.use_lock(%l33_0, Release, 0)
    aie.use_lock(%l33_1, Release, 0)
    aie.next_bd ^end
  ^end:
    aie.end
  }
 }
}
