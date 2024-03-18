//===- test_lock7.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// Fixme: create-locks iterates over maps, so this might fail.

// CHECK-LABEL: module @test_lock5 {
// CHECK:  %0 = aie.tile(5, 5)
// CHECK:  %1 = aie.lock(%0, 0)
// CHECK:  %2 = aie.tile(4, 4)
// CHECK:  %3 = aie.lock(%2, 0)
// CHECK:  %4 = aie.tile(3, 3)
// CHECK:  %5 = aie.lock(%4, 1)
// CHECK:  %6 = aie.lock(%4, 0)
// CHECK:  %7 = aie.buffer(%4) : memref<256xi32>
// CHECK:  %8 = aie.buffer(%2) : memref<256xi32>
// CHECK:  %9 = aie.buffer(%0) : memref<256xi32>
// CHECK:  aiex.token(0) {sym_name = "token0"}
// CHECK:  aiex.token(0) {sym_name = "token1"}
// CHECK:  %10 = aie.mem(%4) {
// CHECK:    aie.use_lock({{.*}}, Acquire, 1)
// CHECK:    aie.dma_bd(%7 : memref<256xi32>) {len = 256 : i32}
// CHECK:    aie.use_lock({{.*}}, Release, 0)
// CHECK:    aie.use_lock({{.*}}, Acquire, 1)
// CHECK:    aie.dma_bd(%7 : memref<256xi32>) {len = 256 : i32}
// CHECK:    aie.use_lock({{.*}}, Release, 0)
// CHECK:    aie.end
// CHECK:  }
// CHECK:  %11 = aie.mem(%2) {
// CHECK:    aie.use_lock(%3, Acquire, 0)
// CHECK:    aie.dma_bd(%8 : memref<256xi32>) {len = 256 : i32}
// CHECK:    aie.use_lock(%3, Release, 1)
// CHECK:    aie.end
// CHECK:  }
// CHECK:  %12 = aie.mem(%0) {
// CHECK:    aie.use_lock(%1, Acquire, 0)
// CHECK:    aie.dma_bd(%9 : memref<256xi32>) {len = 256 : i32}
// CHECK:    aie.use_lock(%1, Release, 1)
// CHECK:    aie.end
// CHECK:  }
// CHECK:  %13 = aie.core(%4) {
// CHECK:    aie.use_lock(%[[Lock1:.*]], Acquire, 0)
// CHECK:    aie.use_lock(%[[Lock2:.*]], Acquire, 0)
// CHECK:    aie.use_lock(%[[Lock1]], Release, 1)
// CHECK:    aie.use_lock(%[[Lock2]], Release, 1)
// CHECK:    aie.end
// CHECK:  }
// CHECK:  %14 = aie.core(%2) {
// CHECK:    aie.use_lock(%3, Acquire, 1)
// CHECK:    aie.use_lock(%3, Release, 0)
// CHECK:    aie.end
// CHECK:  }
// CHECK:  %15 = aie.core(%0) {
// CHECK:    aie.use_lock(%1, Acquire, 1)
// CHECK:    aie.use_lock(%1, Release, 0)
// CHECK:    aie.end
// CHECK:  }
// CHECK:  aie.flow(%4, DMA : 0, %2, DMA : 0)
// CHECK:  aie.flow(%4, DMA : 1, %0, DMA : 0)
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// [Core-Mem] ---> [Core-Mem] (non-neighboring tiles)
//     |---------> [Core-Mem]
// single producer, multipler consumers
module @test_lock5 {
 aie.device(xcvc1902) {
  %t55 = aie.tile(5, 5)
  %t44 = aie.tile(4, 4)
  %t33 = aie.tile(3, 3)

  %buf33 = aie.buffer(%t33) : memref<256xi32>
  %buf44 = aie.buffer(%t44) : memref<256xi32>
  %buf55 = aie.buffer(%t55) : memref<256xi32>

  aiex.token(0) {sym_name = "token0"}
  aiex.token(0) {sym_name = "token1"}

  %m33 = aie.mem(%t33) {
      %dmaSt0 = aie.dma_start(MM2S0, ^bd0, ^dma0)
    ^dma0:
      %dmaSt1 = aie.dma_start("MM2S1", ^bd1, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf33 : memref<256xi32>) { len = 256 : i32 }
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^bd1:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf33 : memref<256xi32>) { len = 256 : i32 }
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m44 = aie.mem(%t44) {
      %dmaSt = aie.dma_start(S2MM0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf44 : memref<256xi32>) { len = 256 : i32 }
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m55 = aie.mem(%t55) {
      %dmaSt = aie.dma_start(S2MM0, ^bd0, ^end)
    ^bd0:
      aiex.useToken @token0(Acquire, 1)
      aie.dma_bd(%buf55 : memref<256xi32>) { len = 256 : i32 }
      aiex.useToken @token0(Release, 2)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %c33 = aie.core(%t33) {
    aiex.useToken @token0(Acquire, 0)
    aiex.useToken @token0(Release, 1)
    aie.end
  }

  %c44 = aie.core(%t44) {
    aiex.useToken @token0(Acquire, 2)
    aiex.useToken @token0(Release, 3)
    aie.end
  }

  %c55 = aie.core(%t55) {
    aiex.useToken @token0(Acquire, 2)
    aiex.useToken @token0(Release, 3)
    aie.end
  }

  aie.flow(%t33, DMA : 0, %t44, DMA : 0)
  aie.flow(%t33, DMA : 1, %t55, DMA : 0)
 }
}
