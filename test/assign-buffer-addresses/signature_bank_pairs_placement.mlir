// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file %s | FileCheck %s
// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file %s | aie-opt --aie-assign-buffer-addresses --split-input-file | FileCheck %s

// Backtrack from placing AB in bank A: AC has no room in bank C.
// CHECK: address = 16384 : i32, mem_bank = 1 : i32, sym_name = "ab"
// CHECK: address = 0 : i32, mem_bank = 0 : i32, sym_name = "ac"
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %p = aie.buffer(%t) {sym_name = "pin", address = 32768 : i32} : memref<16384xi8>
    %ab = aie.buffer(%t) {sym_name = "ab"} : memref<16384xi8, 9>
    %ac = aie.buffer(%t) {sym_name = "ac"} : memref<16384xi8, 10>
  }
}

// -----

// Adjacent allowed banks can hold a spanning buffer.
// CHECK: address = 16384 : i32, mem_bank = 1 : i32, sym_name = "bc_spanning"
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %bc = aie.buffer(%t) {sym_name = "bc_spanning"} : memref<32768xi8, 12>
  }
}

// -----

// An explicit bank narrows a pair; explicit addresses remain unchanged.
// CHECK: mem_bank = 2 : i32, sym_name = "ac_pinned"
// CHECK: address = 49152 : i32, mem_bank = 3 : i32, sym_name = "bd_address"
// CHECK: address = 16352 : i32, mem_bank = 0 : i32, sym_name = "ab_address_spanning"
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %ac = aie.buffer(%t) {sym_name = "ac_pinned", mem_bank = 2 : i32} : memref<64xi8, 10>
    %bd = aie.buffer(%t) {sym_name = "bd_address", address = 49152 : i32, mem_bank = 3 : i32} : memref<64xi8, 13>
    %ab = aie.buffer(%t) {sym_name = "ab_address_spanning", address = 16352 : i32} : memref<64xi8, 9>
  }
}

// -----

// Empty buffers need no free bytes, even when both allowed banks are full.
// CHECK: address = 0 : i32, mem_bank = 0 : i32, sym_name = "empty_ac"
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %a = aie.buffer(%t) {sym_name = "a", address = 0 : i32} : memref<16384xi8>
    %c = aie.buffer(%t) {sym_name = "c", address = 32768 : i32} : memref<16384xi8>
    %b = aie.buffer(%t) {sym_name = "empty_ac"} : memref<0xi8, 10>
  }
}
