//===- signature_bank_conflict_error.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A buffer's address space and its `mem_bank` state the same fact. Honoring
// either one silently would leave Peano scheduling against a bank the buffer is
// not in, which nothing downstream would report.

// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// CHECK: has address space for bank 1 but is pinned to bank 2

module {
  aie.device(npu2) {
    %tile_0_3 = aie.tile(0, 3)
    %disagree = aie.buffer(%tile_0_3) {sym_name = "disagree", mem_bank = 2 : i32}
        : memref<64xi8, 6>
    %core_0_3 = aie.core(%tile_0_3) { aie.end } { stack_size = 1024 : i32 }
  }
}
