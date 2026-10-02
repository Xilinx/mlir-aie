//===- placement_budget.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The placement-budget option bounds the search per tile. The tile below is
// backtracking_dense_pack's, which needs dozens of backtracks: a small budget
// gives up and says so (naming a buffer, even one pinned to a bank), and a
// non-positive budget is rejected up front.

// RUN: aie-opt --aie-assign-buffer-addresses='placement-budget=1000' %s | FileCheck %s --check-prefix=FITS
// RUN: not aie-opt --aie-assign-buffer-addresses='placement-budget=1' %s 2>&1 | FileCheck %s --check-prefix=ONE
// RUN: not aie-opt --aie-assign-buffer-addresses='placement-budget=5' %s 2>&1 | FileCheck %s --check-prefix=FIVE
// RUN: not aie-opt --aie-assign-buffer-addresses='placement-budget=0' %s 2>&1 | FileCheck %s --check-prefix=ZERO

// FITS: sym_name = "buf3"

// ONE: error: 'aie.buffer' op could not be placed
// ONE: note: the search hit its 1-placement budget with arrangements still untried, so a layout may exist that it did not reach (raise it with placement-budget)

// FIVE: error: 'aie.buffer' op could not be placed
// FIVE: note: the search hit its 5-placement budget

// ZERO: error: placement-budget must be positive, got 0

module @placement_budget {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %bankres0 = aie.buffer(%t) { sym_name = "bankres0", mem_bank = 0 : i32 } : memref<512xi8>
    %bankres1 = aie.buffer(%t) { sym_name = "bankres1", mem_bank = 2 : i32 } : memref<768xi8>
    %buf0 = aie.buffer(%t) { sym_name = "buf0" } : memref<21696xi8>
    %buf1 = aie.buffer(%t) { sym_name = "buf1" } : memref<8896xi8>
    %buf2 = aie.buffer(%t) { sym_name = "buf2" } : memref<22976xi8>
    %buf3 = aie.buffer(%t) { sym_name = "buf3" } : memref<8320xi8>
    aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32, data_size = 640 : i32 }
  }
}
