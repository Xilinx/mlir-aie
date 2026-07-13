//===- per_core_slice_lowering_equivalence.divergent.mlir -----*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(%l02, Acquire, %c0_ul0)
    %v = arith.constant 123 : i32
    %i = arith.constant 3 : index
    memref.store %v, %b02[%i] : memref<16xi32>
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%l02, Release, %c1_ul1)
    aie.end
  }
 }
}
