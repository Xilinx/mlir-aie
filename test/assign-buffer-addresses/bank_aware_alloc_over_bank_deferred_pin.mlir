//===- bank_aware_alloc_over_bank_deferred_pin.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Every other allocator test's design succeeds under the FIRST strategy the
// portfolio tries; none of them can tell the difference if the third
// strategy ({bankConstrainedFirst=false, preferBankAligned=false,
// spreadAcrossBanks=true}) were deleted outright. This one can: it was found
// by differentially fuzzing a build with that strategy removed against the
// real allocator (a random search over thousands of designs; this is the
// smallest one that still reproduces the gap), so it genuinely needs that
// specific ordering, not just "some fallback eventually works."
//
// Placing b6 (mem_bank-pinned, 22216 bytes) at the *start* of bank 6 -- which
// is what happens when bank-constrained buffers are placed ahead of
// unconstrained ones regardless of size, as the first two strategies do --
// bisects what would otherwise be one contiguous run spanning bank 3's tail
// through banks 4, 5, 6 and 7 (~315 kB) into two pieces of ~185 kB and
// ~109 kB, neither large enough for b5's 191939-byte unconstrained buffer.
// Trying b6's own bank in isolation doesn't help either: mem_bank is a hard
// constraint, so it cannot be placed anywhere else.
//
// Sorting unconstrained buffers ahead of same-or-smaller bank-pinned ones
// (this strategy) lets the much larger b5 claim that contiguous run first,
// while it is still whole; b6 then finds ample leftover room inside bank 6
// once b5's tail has passed it. b2 and b4 (both pinned via `address`) exist
// only to shape the tile's free space into the specific configuration that
// makes the difference observable.

// RUN: aie-opt --aie-assign-buffer-addresses='alloc-scheme=bank-aware' %s | FileCheck %s

// CHECK: %b2 = aie.buffer(%mem_tile_0_1) {address = 17144 : i32, mem_bank = 0 : i32, sym_name = "b2"} : memref<7808xi8>
// CHECK: %b4 = aie.buffer(%mem_tile_0_1) {address = 194948 : i32, mem_bank = 2 : i32, sym_name = "b4"} : memref<13536xi8>
// CHECK: %b5 = aie.buffer(%mem_tile_0_1) {address = 208484 : i32, mem_bank = 3 : i32, sym_name = "b5"} : memref<191939xi8>
// CHECK: %b6 = aie.buffer(%mem_tile_0_1) {address = 400424 : i32, mem_bank = 6 : i32, sym_name = "b6"} : memref<22216xi8>

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
