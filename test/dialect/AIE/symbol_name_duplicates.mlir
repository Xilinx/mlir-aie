//===- symbol_name_duplicates.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu1) {
    %tile = aie.tile(1, 2)
    %lock0 = aie.lock(%tile, 0) {sym_name = "dup_lock"}
    // expected-error@+1 {{redefinition of symbol named 'dup_lock'}}
    %lock1 = aie.lock(%tile, 1) {sym_name = "dup_lock"}
  }
}

// -----

module {
  aie.device(npu1) {
    %tile = aie.tile(1, 2)
    %buffer0 = aie.buffer(%tile) {sym_name = "dup_buffer"} : memref<16xi32>
    // expected-error@+1 {{redefinition of symbol named 'dup_buffer'}}
    %buffer1 = aie.buffer(%tile) {sym_name = "dup_buffer"} : memref<16xi32>
  }
}

// -----

module {
  aie.device(npu1) {
    %buffer0 = aie.external_buffer {sym_name = "dup_external_buffer"} : memref<16xi32>
    // expected-error@+1 {{redefinition of symbol named 'dup_external_buffer'}}
    %buffer1 = aie.external_buffer {sym_name = "dup_external_buffer"} : memref<16xi32>
  }
}
