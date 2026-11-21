//===- bad_alias.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-assign-buffer-addresses %s

// An alias lands where its buffer and offset put it, so an offset that leaves
// it short of the alignment a core's vector loads need is refused.
module @misaligned {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %pool = aie.buffer(%t) {sym_name = "pool", address = 1024 : i32} : memref<96xbf16>
    // expected-error@+1 {{at byte 2 of buffer "pool" lands at address 0x402, not aligned to the required}}
    %mid = aie.buffer(%t) {sym_name = "pool_at_1", alias_of = @pool, alias_offset = 2 : i32} : memref<32xbf16>
  }
}

// -----

module @past_its_buffer {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %pool = aie.buffer(%t) {sym_name = "pool"} : memref<96xbf16>
    // expected-error@+1 {{alias of 64 bytes at byte 160 runs past @pool's 192}}
    %mid = aie.buffer(%t) {sym_name = "tail", alias_of = @pool, alias_offset = 160 : i32} : memref<32xbf16>
  }
}

// -----

module @on_another_tile {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %u = aie.tile(0, 3)
    %pool = aie.buffer(%t) {sym_name = "pool"} : memref<96xbf16>
    // expected-error@+1 {{alias_of names @pool on another tile}}
    %mid = aie.buffer(%u) {sym_name = "far", alias_of = @pool, alias_offset = 0 : i32} : memref<32xbf16>
  }
}

// -----

module @offset_without_buffer {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    // expected-error@+1 {{alias_of and alias_offset come together}}
    %mid = aie.buffer(%t) {sym_name = "lone", alias_offset = 0 : i32} : memref<32xbf16>
  }
}

// -----

module @alias_of_an_alias {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %pool = aie.buffer(%t) {sym_name = "pool"} : memref<96xbf16>
    %mid = aie.buffer(%t) {sym_name = "mid", alias_of = @pool, alias_offset = 64 : i32} : memref<32xbf16>
    // expected-error@+1 {{alias_of names @mid, itself an alias}}
    %inner = aie.buffer(%t) {sym_name = "inner", alias_of = @mid, alias_offset = 0 : i32} : memref<16xbf16>
  }
}
