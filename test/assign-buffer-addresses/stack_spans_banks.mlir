//===- stack_spans_banks.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A stack placed with `stack_address` may cross bank boundaries, the same way
// the legacy stack at offset zero always could. Rejecting it only because the
// attribute was written would make identical layouts legal or illegal
// depending on how they were spelled.
//
// What a spanning stack cannot do is claim a memory bank: `-aie-stack-addrspace`
// names one, and aiecc omits it here rather than promise Peano's bank-conflict
// model something untrue. `stack_bank` is the attribute that names a bank, and
// it still has to hold its whole stack -- see the error cases below.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses %s | FileCheck %s

// npu2 banks are 16384 bytes, so this covers banks 0, 1 and part of 2.
// CHECK-LABEL: module @spans_three_banks
// CHECK: stack_address = 0 : i32
module @spans_three_banks {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      aie.end
    } { stack_size = 20480 : i32, stack_address = 0 : i32 }
  }
}

// -----

// Smaller than a bank, but starting part-way through one, so it still spans
// two. Size alone does not decide this.
// CHECK-LABEL: module @straddles_one_boundary
// CHECK: stack_address = 16000 : i32
module @straddles_one_boundary {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32, stack_address = 16000 : i32 }
  }
}
