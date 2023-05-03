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

AIE.device(xcvc1902) {
  %t1 = AIE.tile(1, 1)
  %t2 = AIE.tile(4, 4)
  %lock = AIE.lock(%t2, 3) { sym_name = "lock1" }
  // expected-error@-1 {{'AIE.lock' op in Column 4 and Row 4 is accessed from an unreachable tile in Column 1 and Row 1}}
  AIE.core(%t1) {
    AIE.useLock(%lock, "Acquire", 1)
    // expected-note@-1 {{user}}
    AIE.end
  }
}

// -----

AIE.device(xcvc1902) {
  %t = AIE.tile(2, 2)
  %l = AIE.lock(%t, 3)
  AIE.useLock(%l, "Acquire", 1)
  // expected-error@-1 {{'AIE.useLock' op must be used in a core or memory operation.}}
}

// -----

AIE.device(xcvc1902) {
  %t1 = AIE.tile(1, 1)
  %lock = AIE.lock(%t1, -3) { sym_name = "lock1" }
  // expected-error@-1 {{'AIE.lock' op attribute 'lockID' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0}}
  AIE.core(%t1) {
    AIE.useLock(%lock, "Acquire", 1)
    AIE.end
  }
}
