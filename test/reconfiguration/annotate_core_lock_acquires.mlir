//===- annotate_core_lock_acquires.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-annotate-core-lock-acquires --split-input-file --verify-diagnostics %s | FileCheck %s

// Per lock, whether an acquire is greater-equal and each acquire-equal value,
// including a neighbour's lock; a non-blocking acquire leaves no wait.

// CHECK: } {aiex.lock_acquires = [array<i32: 0, 2, 1, 1>, array<i32: 0, 3, 0, 1>, array<i32: 0, 3, 5, 0, 1, 2>]

module {
  aie.device(npu2) {
    %low = aie.tile(0, 2)
    %high = aie.tile(0, 3)
    %neighbour = aie.lock(%low, 1)
    %own = aie.lock(%high, 0)
    %param = aie.lock(%high, 5)
    %skipped = aie.lock(%high, 6)
    %core = aie.core(%high) {
      %c1 = arith.constant 1 : i32
      %c2 = arith.constant 2 : i32
      aie.use_lock(%neighbour, AcquireGreaterEqual, %c1)
      aie.use_lock(%neighbour, AcquireGreaterEqual, %c2)
      aie.use_lock(%own, AcquireGreaterEqual, %c1)
      aie.use_lock(%param, Acquire, %c1)
      aie.use_lock(%param, Acquire, %c2)
      aie.use_lock(%param, Release, %c1)
      aie.use_lock(%skipped, AcquireGreaterEqual, %c1, NonBlocking)
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 0)
    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %buf = memref.alloca() : memref<1xi32>
      %v = memref.load %buf[%c0] : memref<1xi32>
      // expected-error@+1 {{expected the lock value to be a compile-time constant}}
      aie.use_lock(%lock, Acquire, %v)
      aie.end
    }
  }
}
