//===- bank_aware_alloc_gaps.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packing behaviour once the free space in a bank is a set of holes rather
// than one region above a watermark. All modules use an npu2 core tile: 64 kB
// of L1 as 4 banks of 16 kB, a 1024-byte stack, and a 32-byte load/store bus.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// The hole is [1040, 1200), whose first free byte is not bus-aligned. The
// candidate start is rounded up to 1056 *before* testing whether the buffer
// fits, so the hole is used rather than rejected for being misaligned.
// CHECK-LABEL: module @alignment_rounds_up_inside_hole
// CHECK: %fits_after_rounding = aie.buffer(%tile_0_2) {address = 1056 : i32, mem_bank = 0 : i32, sym_name = "fits_after_rounding"} : memref<64xbf16>
module @alignment_rounds_up_inside_hole {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %pin_lo = aie.buffer(%tile_0_2) {address = 1024 : i32, aligned = false, sym_name = "pin_lo"} : memref<8xbf16>
    %pin_hi = aie.buffer(%tile_0_2) {address = 1200 : i32, aligned = false, sym_name = "pin_hi"} : memref<8xbf16>
    %fits_after_rounding = aie.buffer(%tile_0_2) {sym_name = "fits_after_rounding"} : memref<64xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// Two holes in bank 0: a 1024-byte one at [1024, 2048) and a 256-byte one at
// [2080, 2336). A 256-byte buffer takes the tighter, higher hole rather than
// the first one that happens to fit, keeping the larger clean region free for
// a larger buffer.
// CHECK-LABEL: module @best_fit_prefers_tightest_hole
// CHECK: %snug = aie.buffer(%tile_0_2) {address = 2080 : i32, mem_bank = 0 : i32, sym_name = "snug"} : memref<128xbf16>
module @best_fit_prefers_tightest_hole {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %pin_mid = aie.buffer(%tile_0_2) {address = 2048 : i32, sym_name = "pin_mid"} : memref<16xbf16>
    %pin_rest = aie.buffer(%tile_0_2) {address = 2336 : i32, sym_name = "pin_rest"} : memref<7024xbf16>
    %snug = aie.buffer(%tile_0_2) {sym_name = "snug"} : memref<128xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// The buffer exactly fills the [1024, 2048) hole, exercising the boundary of
// the fit test.
// CHECK-LABEL: module @exact_fit_in_hole
// CHECK: %exact = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "exact"} : memref<512xbf16>
module @exact_fit_in_hole {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %pin_rest = aie.buffer(%tile_0_2) {address = 2048 : i32, sym_name = "pin_rest"} : memref<7168xbf16>
    %exact = aie.buffer(%tile_0_2) {sym_name = "exact"} : memref<512xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// Buffers are placed largest first. "large" needs a whole clean 16 kB bank:
// bank 0 cannot hold it because the stack is there, so it takes bank 1
// exactly. Had the four small buffers been placed first they would have
// spread round-robin over every bank and dirtied all four, leaving "large" no
// bank-contained home and forcing it to straddle a bank boundary.
// CHECK-LABEL: module @largest_first
// CHECK: %large = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "large"} : memref<8192xbf16>
module @largest_first {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %small0 = aie.buffer(%tile_0_2) {sym_name = "small0"} : memref<16xbf16>
    %small1 = aie.buffer(%tile_0_2) {sym_name = "small1"} : memref<16xbf16>
    %small2 = aie.buffer(%tile_0_2) {sym_name = "small2"} : memref<16xbf16>
    %small3 = aie.buffer(%tile_0_2) {sym_name = "small3"} : memref<16xbf16>
    %large = aie.buffer(%tile_0_2) {sym_name = "large"} : memref<8192xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// A zero-sized buffer is degenerate but legal; it must not upset the
// occupancy bookkeeping. It still takes a genuinely free address when one
// exists, rather than being parked on top of another buffer.
// CHECK-LABEL: module @zero_sized_buffer
// CHECK: %empty = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "empty"} : memref<0xi32>
// CHECK: %normal = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "normal"} : memref<16xbf16>
module @zero_sized_buffer {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %empty = aie.buffer(%tile_0_2) {sym_name = "empty"} : memref<0xi32>
    %normal = aie.buffer(%tile_0_2) {sym_name = "normal"} : memref<16xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// Every bank is exactly full, so there is no hole anywhere. A zero-sized
// buffer covers no bytes, so it is still placeable: it must not fail
// allocation, nor be reported as overlapping the buffer or the stack whose
// address it shares.
// CHECK-LABEL: module @zero_sized_buffer_on_full_tile
// CHECK: %empty = aie.buffer(%tile_0_2) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "empty"} : memref<0xi32>
module @zero_sized_buffer_on_full_tile {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %b0 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "b0"} : memref<15360xi8>
    %b1 = aie.buffer(%tile_0_2) {address = 16384 : i32, sym_name = "b1"} : memref<16384xi8>
    %b2 = aie.buffer(%tile_0_2) {address = 32768 : i32, sym_name = "b2"} : memref<16384xi8>
    %b3 = aie.buffer(%tile_0_2) {address = 49152 : i32, sym_name = "b3"} : memref<16384xi8>
    %empty = aie.buffer(%tile_0_2) {sym_name = "empty"} : memref<0xi32>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}
