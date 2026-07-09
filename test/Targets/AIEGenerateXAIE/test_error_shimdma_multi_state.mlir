//===- test_error_shimdma_multi_state.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-translate --aie-generate-xaie %s 2>&1 | FileCheck %s
// CHECK: acquires/releases the lock in a DMA block from/to multiple states.

module @test_error_shimdma_multi_state {
 aie.device(xcvc1902) {
  %t30 = aie.tile(3, 0)
  %l30_0 = aie.lock(%t30, 0)
  aie.shim_dma(%t30) {
    aie.dma_start(MM2S, 0, ^bb1, ^end)
  ^bb1:
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(%l30_0, Acquire, %c0_ul0)
    // This should fail because only one lock can be used in a ShimBd
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%l30_0, Acquire, %c1_ul1)
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%l30_0, Release, %c0_ul2)
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(%l30_0, Release, %c1_ul3)
    aie.next_bd ^end
  ^end:
    aie.end
  }
 }
}
