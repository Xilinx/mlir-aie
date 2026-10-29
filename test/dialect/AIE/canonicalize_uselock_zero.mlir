//===- canonicalize_uselock_zero.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --canonicalize %s | FileCheck %s

// Acquiring 0 is a no-op on a core, but in a buffer descriptor it fills the
// acquire field the release beside it is paired with.

// CHECK-LABEL: module @zero_acquire
// CHECK:       aie.core
// CHECK-NOT:     AcquireGreaterEqual
// CHECK:         aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK:       aie.memtile_dma
// CHECK:         aie.use_lock(%[[FULL:.*]], AcquireGreaterEqual, %c0_i32)
// CHECK-NEXT:    aie.dma_bd
// CHECK-NEXT:    aie.use_lock(%{{.*}}, Release, %c2_i32)
module @zero_acquire {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %lock = aie.lock(%core) {init = 1 : i32}
    %buf = aie.buffer(%mem) : memref<32xi32>
    %free = aie.lock(%mem) {init = 4 : i32}
    %full = aie.lock(%mem) {init = 0 : i32}
    %c = aie.core(%core) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aie.use_lock(%lock, AcquireGreaterEqual, %c0)
      aie.use_lock(%lock, Release, %c1)
      aie.end
    }
    %m = aie.memtile_dma(%mem) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:
      %c0 = arith.constant 0 : i32
      %c2 = arith.constant 2 : i32
      aie.use_lock(%full, AcquireGreaterEqual, %c0)
      aie.dma_bd(%buf : memref<32xi32> offset = 16 len = 16)
      aie.use_lock(%free, Release, %c2)
      aie.next_bd ^bb1
    ^bb2:
      aie.end
    }
  }
}
