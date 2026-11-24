// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A bank pin no placement can satisfy is refused when the buffer is built,
// placed or not, rather than by buffer addressing after placement.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    // expected-error @+1 {{has address space for bank 1 but is pinned to bank 0. These constraints disagree}}
    %b = aie.buffer(%p) {sym_name = "b", mem_bank = 0 : i32} : memref<64xi8, 6>
    aie.core(%p) { aie.end }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    // expected-error @+1 {{has address space for bank 0 or 2 but is pinned to bank 1}}
    %b = aie.buffer(%p) {sym_name = "b", mem_bank = 1 : i32} : memref<64xi8, 10>
    aie.core(%p) { aie.end }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    // expected-error @+1 {{mem_bank attribute value is out of range}}
    %b = aie.buffer(%p) {sym_name = "b", mem_bank = 7 : i32} : memref<64xi8>
    aie.core(%p) { aie.end }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    // expected-error @+1 {{mem_bank attribute value is out of range}}
    %b = aie.buffer(%t) {sym_name = "b", mem_bank = 4 : i32} : memref<64xi8>
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<MemTile>(?, ?)
    // expected-error @+1 {{mem_bank attribute value is out of range}}
    %b = aie.buffer(%p) {sym_name = "b", mem_bank = -1 : i32} : memref<64xi8>
  }
}

// -----

// A mem tile has 8 banks, so bank 7 is a pin it can hold.
module {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<MemTile>(?, ?)
    %b = aie.buffer(%p) {sym_name = "b", mem_bank = 7 : i32} : memref<64xi8>
  }
}
