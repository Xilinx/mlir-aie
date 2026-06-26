//===- per_core_slice_lowering_equivalence.slice.mlir ---------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Stripped slice for core (0, 2), produced the way compileCores produces it:
// @device1 only (@device2 removed), the runtime sequence removed, and the other
// core's aie.core op removed. The other core's tile, buffer, and lock are kept,
// because compileCores erases only the aie.core ops. Driven by the RUN lines in
// per_core_slice_lowering_equivalence.mlir.
module @slice {
 aie.device(npu1_2col) @device1 {
  %t02 = aie.tile(0, 2)
  %t12 = aie.tile(1, 2)
  %l02 = aie.lock(%t02, 0)
  %b02 = aie.buffer(%t02) { sym_name = "a02" } : memref<16xi32>
  %l12 = aie.lock(%t12, 0)
  %b12 = aie.buffer(%t12) { sym_name = "a12" } : memref<16xi32>
  %c02 = aie.core(%t02) {
    aie.use_lock(%l02, Acquire, 0)
    %v = arith.constant 7 : i32
    %i = arith.constant 3 : index
    memref.store %v, %b02[%i] : memref<16xi32>
    aie.use_lock(%l02, Release, 1)
    aie.end
  }
 }
}
