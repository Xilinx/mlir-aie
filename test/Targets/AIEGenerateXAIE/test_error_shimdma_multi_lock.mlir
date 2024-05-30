//===- test_error_shimdma_multi_lock.mlir ----------------------*- MLIR -*-===//
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

module @test_error_shimdma_multi_lock {
 aie.device(xcvc1902) {
  %t30 = aie.tile(3, 0)
  %l30_0 = aie.lock(%t30, 0)
  %l30_1 = aie.lock(%t30, 1)
  aie.shim_dma(%t30) {
    aie.dma_start(MM2S, 0, ^bb1, ^end)
  ^bb1:
    aie.use_lock(%l30_0, Acquire, 1)
    // This should fail because only one state can be acquired in a ShimBd
    aie.use_lock(%l30_1, Acquire, 1)
    aie.use_lock(%l30_0, Release, 0)
    aie.use_lock(%l30_1, Release, 0)
    aie.next_bd ^end
  ^end:
    aie.end
  }
 }
}
