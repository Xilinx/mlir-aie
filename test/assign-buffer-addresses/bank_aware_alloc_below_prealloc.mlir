//===- bank_aware_alloc_below_prealloc.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Free space *below* a buffer with an explicit address must be usable. A
// per-bank "next free address" watermark strands it, because a fixed-address
// buffer pushes the watermark past itself.
//
// The hardware imposes no natural-size or bank alignment requirement on a
// buffer, so nothing here is alignment-driven: a buffer only rounds up to the
// tile's load/store bus width.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// Core tile on npu2: 64 kB of L1 as 4 banks of 16 kB. Banks 1, 2 and 3 are
// pinned full. Bank 0 holds the 1024-byte stack, a 32-byte buffer pinned at
// 8192 and a filler running from 8224 to the top of the bank, so the only
// space left anywhere on the tile is the 7168-byte hole at [1024, 8192) --
// below the pin. The 4096-byte buffer has to land there.
// CHECK-LABEL: module @hole_below_pin_core
// CHECK: %unplaced = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "unplaced"} : memref<2048xbf16>
module @hole_below_pin_core {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %pin_bank1 = aie.buffer(%tile_0_2) {address = 16384 : i32, sym_name = "pin_bank1"} : memref<8192xbf16>
    %pin_bank2 = aie.buffer(%tile_0_2) {address = 32768 : i32, sym_name = "pin_bank2"} : memref<8192xbf16>
    %pin_bank3 = aie.buffer(%tile_0_2) {address = 49152 : i32, sym_name = "pin_bank3"} : memref<8192xbf16>
    %pin_bank0 = aie.buffer(%tile_0_2) {address = 8192 : i32, sym_name = "pin_bank0"} : memref<16xbf16>
    %pin_bank0_hi = aie.buffer(%tile_0_2) {address = 8224 : i32, sym_name = "pin_bank0_hi"} : memref<4080xbf16>
    %unplaced = aie.buffer(%tile_0_2) {sym_name = "unplaced"} : memref<2048xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// The same property on a memtile, reached through the mem_bank path: bank 0
// of this 512 kB memtile is 64 kB, its upper half is pinned, and the buffer
// requesting bank 0 has to use the 32768-byte hole below that pin.
// CHECK-LABEL: module @hole_below_pin_memtile
// CHECK: %wants_bank0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "wants_bank0"} : memref<8192xi32>
module @hole_below_pin_memtile {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %pin_upper_half = aie.buffer(%mem_tile_0_1) {address = 32768 : i32, sym_name = "pin_upper_half"} : memref<8192xi32>
    %wants_bank0 = aie.buffer(%mem_tile_0_1) {mem_bank = 0 : i32, sym_name = "wants_bank0"} : memref<8192xi32>
    aie.memtile_dma(%mem_tile_0_1) {
      aie.end
    }
  }
}
