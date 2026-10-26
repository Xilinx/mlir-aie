// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --aie-assign-buffer-addresses %s | aie-opt --aie-assign-buffer-addresses | FileCheck %s

// Occupy the first allowed bank: each resource pair must use its other bank,
// not whichever unconstrained bank the allocator happens to prefer.
// CHECK: mem_bank = 1 : i32, sym_name = "ab"
// CHECK: mem_bank = 2 : i32, sym_name = "ac"
// CHECK: mem_bank = 3 : i32, sym_name = "ad"
// CHECK: mem_bank = 2 : i32, sym_name = "bc"
// CHECK: mem_bank = 3 : i32, sym_name = "bd"
// CHECK: mem_bank = 3 : i32, sym_name = "cd"
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 2)
    %p0 = aie.buffer(%t0) {sym_name = "pin0", mem_bank = 0 : i32} : memref<16384xi8>
    %ab = aie.buffer(%t0) {sym_name = "ab"} : memref<64xi8, 9>
    %t1 = aie.tile(1, 2)
    %p1 = aie.buffer(%t1) {sym_name = "pin1", mem_bank = 0 : i32} : memref<16384xi8>
    %ac = aie.buffer(%t1) {sym_name = "ac"} : memref<64xi8, 10>
    %t2 = aie.tile(2, 2)
    %p2 = aie.buffer(%t2) {sym_name = "pin2", mem_bank = 0 : i32} : memref<16384xi8>
    %ad = aie.buffer(%t2) {sym_name = "ad"} : memref<64xi8, 11>
    %t3 = aie.tile(3, 2)
    %p3 = aie.buffer(%t3) {sym_name = "pin3", mem_bank = 1 : i32} : memref<16384xi8>
    %bc = aie.buffer(%t3) {sym_name = "bc"} : memref<64xi8, 12>
    %t4 = aie.tile(4, 2)
    %p4 = aie.buffer(%t4) {sym_name = "pin4", mem_bank = 1 : i32} : memref<16384xi8>
    %bd = aie.buffer(%t4) {sym_name = "bd"} : memref<64xi8, 13>
    %t5 = aie.tile(5, 2)
    %p5 = aie.buffer(%t5) {sym_name = "pin5", mem_bank = 2 : i32} : memref<16384xi8>
    %cd = aie.buffer(%t5) {sym_name = "cd"} : memref<64xi8, 14>
  }
}
