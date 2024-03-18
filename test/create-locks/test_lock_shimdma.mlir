//===- test_lock3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: andrab
// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABELs: module @test_lock_shimdma  {
// CHECK:   aiex.token(0) {sym_name = "token0"}
// CHECK:   %0 = aie.external_buffer : memref<256xi32>
// CHECK:   %1 = aie.tile(6, 0)
// CHECK:   %2 = aie.lock(%1, 0)
// CHECK:   %3 = aie.core(%1)  {
// CHECK:     aie.use_lock(%2, Acquire, 1)
// CHECK:     aie.use_lock(%2, Release, 0)
// CHECK:     aie.end
// CHECK:   }
// CHECK:   %4 = aie.shim_dma(%1)  {
// CHECK:     %10 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:   ^bb1:  // pred: ^bb0
// CHECK:     aie.use_lock(%2, Acquire, 0)
// CHECK:     aie.dma_bd(%0 : memref<256xi32>) {len = 256 : i32}
// CHECK:     aie.use_lock(%2, Release, 1)
// CHECK:     aie.next_bd ^bb2
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:     aie.end
// CHECK:   }
// CHECK:   %5 = aie.tile(3, 3)
// CHECK:   %6 = aie.lock(%5, 0)
// CHECK:   %7 = aie.buffer(%5) : memref<256xi32>
// CHECK:   %8 = aie.core(%5)  {
// CHECK:     aie.use_lock(%6, Acquire, 0)
// CHECK:     aie.use_lock(%6, Release, 1)
// CHECK:     aie.end
// CHECK:   }
// CHECK:   %9 = aie.mem(%5)  {
// CHECK:     %10 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:   ^bb1:  // pred: ^bb0
// CHECK:     aie.use_lock(%6, Acquire, 1)
// CHECK:     aie.dma_bd(%7 : memref<256xi32>) {len = 256 : i32}
// CHECK:     aie.use_lock(%6, Release, 0)
// CHECK:     aie.next_bd ^bb2
// CHECK:   ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:     aie.end
// CHECK:   }
// CHECK:   aie.flow(%5, DMA : 0, %1, DMA : 0)
// CHECK: }

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [ShimDMA] (non-neighboring tiles)
// single producer, single consumer
module @test_lock_shimdma {
 aie.device(xcvc1902) {
  aiex.token(0) {sym_name = "token0"}
  %buf_ext = aie.external_buffer : memref<256xi32>

  %t60 = aie.tile(6, 0)
  %c60 = aie.core(%t60) {
    // TODO: This represents the token uses on the host CPU. A representation of
    // the host CPU in MLIR might be a better place for holding this.
    aiex.useToken @token0(Acquire, 2)
    aiex.useToken @token0(Release, 3)
    aie.end
  }
  %m60 = aie.shim_dma(%t60) {
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf_ext : memref<256xi32>) { len = 256 : i32 }
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %t33 = aie.tile(3, 3)
  %buf33 = aie.buffer(%t33) : memref<256xi32>
  %c33 = aie.core(%t33) {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token0(Release, 1)
    aie.end
  }
  %m33 = aie.mem(%t33) {
      %dmaSt = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf33 : memref<256xi32>) { len = 256 : i32 }
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  aie.flow(%t33, DMA : 0, %t60, DMA : 0)
 }
}
