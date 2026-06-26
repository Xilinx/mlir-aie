//===- per_core_slice_lowering_equivalence.divergent.mlir -----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Negative fixture for per_core_slice_lowering_equivalence.mlir. Same as the
// stripped slice, except the target core stores a different value (123 instead
// of 7), so its lowering diverges. A "not FileCheck" RUN asserts this does NOT
// match the golden, which proves the golden actually pins the target core's
// computation (guards the equivalence check against silently going vacuous).
module @slice_divergent {
 aie.device(npu1_2col) @device1 {
  %t02 = aie.tile(0, 2)
  %t12 = aie.tile(1, 2)
  %l02 = aie.lock(%t02, 0)
  %b02 = aie.buffer(%t02) { sym_name = "a02" } : memref<16xi32>
  %l12 = aie.lock(%t12, 0)
  %b12 = aie.buffer(%t12) { sym_name = "a12" } : memref<16xi32>
  %c02 = aie.core(%t02) {
    aie.use_lock(%l02, Acquire, 0)
    %v = arith.constant 123 : i32
    %i = arith.constant 3 : index
    memref.store %v, %b02[%i] : memref<16xi32>
    aie.use_lock(%l02, Release, 1)
    aie.end
  }
 }
}
