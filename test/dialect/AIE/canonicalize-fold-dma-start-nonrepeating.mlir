//===- canonicalize-fold-dma-start-nonrepeating.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A cyclic BD chain whose blocks are pairwise distinct has nothing to fold:
// FoldDMAStartOp must fail rather than report success() on unchanged IR,
// which test-convergence turns into a non-convergence error.

// RUN: aie-opt --canonicalize="test-convergence max-iterations=2" %s | FileCheck %s

// CHECK: aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK: ^bb1:
// CHECK:   aie.dma_bd(%{{.*}} : memref<256xi32> offset = 0 len = 256)
// CHECK:   aie.next_bd ^bb2
// CHECK: ^bb2:
// CHECK:   aie.dma_bd(%{{.*}} : memref<256xi32> offset = 0 len = 128)
// CHECK:   aie.next_bd ^bb1
// CHECK: ^bb3:
// CHECK:   aie.end

module @test {
  %t1 = aie.tile(1, 1)
  %buf_0 = aie.buffer(%t1) { sym_name = "buf_0" } : memref<256xi32>
  %buf_1 = aie.buffer(%t1) { sym_name = "buf_1" } : memref<256xi32>
  %lock_0 = aie.lock(%t1, 0)

  %mem = aie.mem(%t1) {
    %start = aie.dma_start("MM2S", 0, ^bd0, ^end)
  ^bd0:
    %c1 = arith.constant 1 : i32
    aie.use_lock(%lock_0, Acquire, %c1)
    aie.dma_bd(%buf_0 : memref<256xi32> offset = 0 len = 256)
    %c0 = arith.constant 0 : i32
    aie.use_lock(%lock_0, Release, %c0)
    aie.next_bd ^bd1
  ^bd1:
    %c1b = arith.constant 1 : i32
    aie.use_lock(%lock_0, Acquire, %c1b)
    aie.dma_bd(%buf_1 : memref<256xi32> offset = 0 len = 128)
    %c0b = arith.constant 0 : i32
    aie.use_lock(%lock_0, Release, %c0b)
    aie.next_bd ^bd0
  ^end:
    aie.end
  }
}
