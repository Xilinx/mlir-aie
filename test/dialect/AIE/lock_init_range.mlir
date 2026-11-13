//===- lock_init_range.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A lock holds 0 to the target's maximum lock value: 63 on AIE2 and AIE2P, 1 on
// AIE1.

aie.device(npu2) {
  %t = aie.tile(0, 2)
  %ok = aie.lock(%t) {init = 63 : i32}
  // expected-error@+1 {{init value 64 is outside the lock's range [0, 63]}}
  %l = aie.lock(%t) {init = 64 : i32}
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 2)
  // expected-error@+1 {{init value -1 is outside the lock's range [0, 63]}}
  %l = aie.lock(%t) {init = -1 : i32}
}

// -----

aie.device(xcvc1902) {
  %t = aie.tile(1, 3)
  %ok = aie.lock(%t, 0) {init = 1 : i32}
  // expected-error@+1 {{init value 2 is outside the lock's range [0, 1]}}
  %l = aie.lock(%t, 1) {init = 2 : i32}
}
