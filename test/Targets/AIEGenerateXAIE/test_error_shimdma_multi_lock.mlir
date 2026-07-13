//===- test_error_shimdma_multi_lock.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-translate --aie-generate-xaie %s 2>&1 | FileCheck %s
// CHECK: used in a DMA block that have multiple locks.

module @test_error_shimdma_multi_lock {
 aie.device(xcvc1902) {
  %t30 = aie.tile(3, 0)
  %l30_0 = aie.lock(%t30, 0)
  %l30_1 = aie.lock(%t30, 1)
  aie.shim_dma(%t30) {
    aie.dma_start(MM2S, 0, ^bb1, ^end)
  ^bb1:
    %c1_ul0 = arith.constant 1 : i32
    aie.use_lock(%l30_0, Acquire, %c1_ul0)
    // This should fail because only one state can be acquired in a ShimBd
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%l30_1, Acquire, %c1_ul1)
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%l30_0, Release, %c0_ul2)
    %c0_ul3 = arith.constant 0 : i32
    aie.use_lock(%l30_1, Release, %c0_ul3)
    aie.next_bd ^end
  ^end:
    aie.end
  }
 }
}
