//===- per_core_slice_lowering_equivalence.wrongcore.mlir -----*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Negative fixture for per_core_slice_lowering_equivalence.mlir. The slice keeps
// the wrong core: the other core (1, 2) instead of the target (0, 2). Lowering
// for tilecol=0 tilerow=2 then yields no @core_0_2, so a "not FileCheck" RUN
// asserts it does NOT match the golden, proving the golden pins which core is
// lowered.
module @slice_wrongcore {
 aie.device(npu1_2col) @device1 {
  %t12 = aie.tile(1, 2)
  %l12 = aie.lock(%t12, 0)
  %b12 = aie.buffer(%t12) { sym_name = "a12" } : memref<16xi32>
  %c12 = aie.core(%t12) {
    aie.use_lock(%l12, Acquire, 0)
    %v = arith.constant 99 : i32
    %i = arith.constant 5 : index
    memref.store %v, %b12[%i] : memref<16xi32>
    aie.use_lock(%l12, Release, 1)
    aie.end
  }
 }
}
