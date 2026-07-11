//===- uselock_dynamic_value.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Roundtrip test for runtime (SSA) lock values on AIE2. The lock value is a
// general-purpose i32 register operand rather than a static attribute.

// RUN: aie-opt %s | FileCheck %s

module {
  aie.device(npu1) {
    %tile = aie.tile(1, 2)
    %lock = aie.lock(%tile, 0) {init = 1 : i32}
    %core = aie.core(%tile) {
      %v = arith.constant 2 : i32
      // CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
      aie.use_lock(%lock, AcquireGreaterEqual, %v)
      // CHECK: aie.use_lock(%{{.*}}, Release, %{{.*}})
      aie.use_lock(%lock, Release, %v)
      aie.end
    }
  }
}
