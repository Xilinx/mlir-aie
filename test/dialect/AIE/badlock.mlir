//===- badlock.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

aie.device(xcvc1902) {
  %t1 = aie.tile(1, 1)
  %t2 = aie.tile(4, 4)
  %lock = aie.lock(%t2, 3) { sym_name = "lock1" }
  // expected-error@-1 {{'aie.lock' op in Column 4 and Row 4 is accessed from an unreachable tile in Column 1 and Row 1}}
  aie.core(%t1) {
    aie.use_lock(%lock, "Acquire", 1)
    // expected-note@-1 {{user}}
    aie.end
  }
}

// -----

aie.device(xcvc1902) {
  %t = aie.tile(2, 2)
  %l = aie.lock(%t, 3)
  aie.use_lock(%l, "Acquire", 1)
  // expected-error@-1 {{'aie.use_lock' op must be used in a core or memory operation.}}
}

// -----

aie.device(xcvc1902) {
  %t1 = aie.tile(1, 1)
  %lock = aie.lock(%t1, -3) { sym_name = "lock1" }
  // expected-error@-1 {{'aie.lock' op attribute 'lockID' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0}}
  aie.core(%t1) {
    aie.use_lock(%lock, "Acquire", 1)
    aie.end
  }
}

// -----

aie.device(xcvc1902) {
  %t = aie.tile(3, 3)
  %l = aie.lock(%t, 0)
  aie.core(%t) {
    // expected-error@+1 {{'aie.use_lock' op AcquireGreaterEqual is not supported in AIE1.}}
    aie.use_lock(%l, AcquireGreaterEqual, 1)
    aie.end
  }
}