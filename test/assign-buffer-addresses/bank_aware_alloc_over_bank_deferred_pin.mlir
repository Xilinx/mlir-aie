//===- bank_aware_alloc_over_bank_deferred_pin.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A bank-pinned buffer is placed before the unconstrained ones, so where it
// lands inside its bank decides whether a large unconstrained buffer still
// fits. findLeastFragmentingGap picks the position that leaves the largest
// single run behind, and this design is the smallest one that can observe the
// difference. A random search over thousands of designs produced it.
//
// b6 is pinned to bank 6 and needs 22216 bytes. Its bank sits in the middle of
// one contiguous run that spans bank 3's tail through banks 4, 5, 6 and 7,
// about 315 kB. Placing b6 at the start of bank 6 splits that run into pieces
// of about 185 kB and 109 kB, and b5 needs 191939 contiguous bytes. mem_bank is
// a hard constraint, so b6 cannot move to another bank.
//
// Placing b6 flush against the far end of its bank keeps the run whole below
// it, and b5 fits. b2 and b4, both pinned by `address`, shape the tile's free
// space into that configuration.

// RUN: aie-opt --aie-assign-buffer-addresses='alloc-scheme=bank-aware' %s | FileCheck %s

// CHECK: %b2 = aie.buffer(%mem_tile_0_1) {address = 17144 : i32, mem_bank = 0 : i32, sym_name = "b2"} : memref<7808xi8>
// CHECK: %b4 = aie.buffer(%mem_tile_0_1) {address = 194948 : i32, mem_bank = 2 : i32, sym_name = "b4"} : memref<13536xi8>
// CHECK: %b5 = aie.buffer(%mem_tile_0_1) {address = 208484 : i32, mem_bank = 3 : i32, sym_name = "b5"} : memref<191939xi8>
// b6 slides to the far end of bank 6, the position that leaves the free space
// below it in one run.
// CHECK: %b6 = aie.buffer(%mem_tile_0_1) {address = 436536 : i32, mem_bank = 6 : i32, sym_name = "b6"} : memref<22216xi8>

module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    %b2 = aie.buffer(%t) {sym_name = "b2", address = 17144 : i32} : memref<7808xi8>
    %b4 = aie.buffer(%t) {sym_name = "b4", address = 194948 : i32} : memref<13536xi8>
    %b5 = aie.buffer(%t) {sym_name = "b5"} : memref<191939xi8>
    %b6 = aie.buffer(%t) {sym_name = "b6", mem_bank = 6 : i32} : memref<22216xi8>
    aie.memtile_dma(%t) { aie.end }
  }
}
