//===- badlock.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s -split-input-file -verify-diagnostics

aie.device(xcvc1902) {
  %t1 = aie.tile(1, 1)
  %t2 = aie.tile(4, 4)
  %lock = aie.lock(%t2, 3) { sym_name = "lock1" }
  // expected-error@-1 {{'aie.lock' op in Column 4 and Row 4 is accessed from an unreachable tile in Column 1 and Row 1}}
  aie.core(%t1) {
    %c1_ul0 = arith.constant 1 : i32
    aie.use_lock(%lock, "Acquire", %c1_ul0)
    // expected-note@-1 {{user}}
    aie.end
  }
}

// -----

aie.device(xcvc1902) {
  %t = aie.tile(2, 2)
  %l = aie.lock(%t, 3)
  %c1_ul1 = arith.constant 1 : i32
  aie.use_lock(%l, "Acquire", %c1_ul1)
  // expected-error@-1 {{'aie.use_lock' op must be used in a core or memory operation.}}
}

// -----

aie.device(xcvc1902) {
  %t1 = aie.tile(1, 1)
  %lock = aie.lock(%t1, -3) { sym_name = "lock1" }
  // expected-error@-1 {{'aie.lock' op attribute 'lockID' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0}}
  aie.core(%t1) {
    %c1_ul2 = arith.constant 1 : i32
    aie.use_lock(%lock, "Acquire", %c1_ul2)
    aie.end
  }
}

// -----

aie.device(xcvc1902) {
  %t = aie.tile(3, 3)
  %l = aie.lock(%t, 0)
  aie.core(%t) {
    %c1_ul3 = arith.constant 1 : i32
    // expected-error@+1 {{'aie.use_lock' op AcquireGreaterEqual is not supported in AIE1.}}
    aie.use_lock(%l, AcquireGreaterEqual, %c1_ul3)
    aie.end
  }
}