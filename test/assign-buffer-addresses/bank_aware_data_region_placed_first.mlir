//===- bank_aware_data_region_placed_first.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The core's own sections compete with the buffers for the same memory, so
// placement treats them as one more extent to fit.
//
// Both cases below hold six 8192-byte buffers on an npu2 core tile: 65536 bytes
// over four banks of 16384, with a 1024-byte stack. A core needing 12000 bytes
// of .data, .rodata and .bss fits, because 6 * 8192 + 12000 is 61152 and 64512
// bytes are free. Whether placement finds that packing depends on whether the
// core declares the need.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// Without a declared size there is no extent to place, so the buffers take
// their addresses first and the core gets whatever run survives. Each buffer
// fills a bank to its boundary, and no run of 12000 bytes remains. The core
// link reports the shortfall.
// CHECK-LABEL: module @nothing_declared
// CHECK-NOT: core_data
// CHECK: %b0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "b0"} : memref<8192xi8>
// CHECK: %b5 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "b5"} : memref<8192xi8>
module @nothing_declared {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %b0 = aie.buffer(%tile_0_2) {sym_name = "b0"} : memref<8192xi8>
    %b1 = aie.buffer(%tile_0_2) {sym_name = "b1"} : memref<8192xi8>
    %b2 = aie.buffer(%tile_0_2) {sym_name = "b2"} : memref<8192xi8>
    %b3 = aie.buffer(%tile_0_2) {sym_name = "b3"} : memref<8192xi8>
    %b4 = aie.buffer(%tile_0_2) {sym_name = "b4"} : memref<8192xi8>
    %b5 = aie.buffer(%tile_0_2) {sym_name = "b5"} : memref<8192xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// The same design with the size declared. `data_size` becomes a core_data
// buffer, placement sorts it among the rest by size, it takes the bottom of the
// tile, the six buffers move up one slot, and all seven extents fit.
// CHECK-LABEL: module @data_size_declared
// CHECK: %b0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "b0"} : memref<8192xi8>
// CHECK: %b5 = aie.buffer(%tile_0_2) {address = 57344 : i32, mem_bank = 3 : i32, sym_name = "b5"} : memref<8192xi8>
// CHECK: %core_data_0_2 = aie.buffer(%tile_0_2) {address = 1024 : i32, core_data, mem_bank = 0 : i32, sym_name = "core_data_0_2"} : memref<12000xi8>
module @data_size_declared {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %b0 = aie.buffer(%tile_0_2) {sym_name = "b0"} : memref<8192xi8>
    %b1 = aie.buffer(%tile_0_2) {sym_name = "b1"} : memref<8192xi8>
    %b2 = aie.buffer(%tile_0_2) {sym_name = "b2"} : memref<8192xi8>
    %b3 = aie.buffer(%tile_0_2) {sym_name = "b3"} : memref<8192xi8>
    %b4 = aie.buffer(%tile_0_2) {sym_name = "b4"} : memref<8192xi8>
    %b5 = aie.buffer(%tile_0_2) {sym_name = "b5"} : memref<8192xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_size = 12000 : i32}
  }
}
