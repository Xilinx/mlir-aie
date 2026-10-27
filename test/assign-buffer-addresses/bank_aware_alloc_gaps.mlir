//===- bank_aware_alloc_gaps.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packing behavior when the free space in a bank is a set of holes. All
// modules use an npu2 core tile: 64 kB of L1 as 4 banks of 16 kB, a 1024-byte
// stack, a 32-byte load/store bus, and a 64-byte alignment for any buffer large
// enough to hold a full-width vector (see getRequiredAlignBits).

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses %s | FileCheck %s

// The hole is [1040, 1216), whose first free byte is misaligned. The candidate
// start rounds up to 1088 *before* the fit test, so the buffer uses the hole.
// CHECK-LABEL: module @alignment_rounds_up_inside_hole
// CHECK: aie.buffer @pin_lo(%tile_0_2) {address = 1024 : i32, aligned = false, mem_bank = 0 : i32} : memref<8xbf16>
// CHECK: aie.buffer @pin_hi(%tile_0_2) {address = 1216 : i32, aligned = false, mem_bank = 0 : i32} : memref<8xbf16>
// CHECK: aie.buffer @fits_after_rounding(%tile_0_2) {address = 1088 : i32, mem_bank = 0 : i32} : memref<64xbf16>
module @alignment_rounds_up_inside_hole {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @pin_lo(%tile_0_2) {address = 1024 : i32, aligned = false} : memref<8xbf16>
    aie.buffer @pin_hi(%tile_0_2) {address = 1216 : i32, aligned = false} : memref<8xbf16>
    aie.buffer @fits_after_rounding(%tile_0_2) : memref<64xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// Two holes in bank 0: a 1024-byte one at [1024, 2048) and a 256-byte one at
// [2112, 2368). A 256-byte buffer takes the tighter, higher hole, which keeps
// the larger clean region free for a larger buffer.
// CHECK-LABEL: module @best_fit_prefers_tightest_hole
// CHECK: aie.buffer @pin_mid(%tile_0_2) {address = 2048 : i32, mem_bank = 0 : i32} : memref<32xbf16>
// CHECK: aie.buffer @pin_rest(%tile_0_2) {address = 2368 : i32, mem_bank = 0 : i32} : memref<7008xbf16>
// CHECK: aie.buffer @snug(%tile_0_2) {address = 2112 : i32, mem_bank = 0 : i32} : memref<128xbf16>
module @best_fit_prefers_tightest_hole {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @pin_mid(%tile_0_2) {address = 2048 : i32} : memref<32xbf16>
    aie.buffer @pin_rest(%tile_0_2) {address = 2368 : i32} : memref<7008xbf16>
    aie.buffer @snug(%tile_0_2) : memref<128xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// The buffer exactly fills the [1024, 2048) hole, exercising the boundary of
// the fit test.
// CHECK-LABEL: module @exact_fit_in_hole
// CHECK: aie.buffer @pin_rest(%tile_0_2) {address = 2048 : i32, mem_bank = 0 : i32} : memref<7168xbf16>
// CHECK: aie.buffer @exact(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32} : memref<512xbf16>
module @exact_fit_in_hole {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @pin_rest(%tile_0_2) {address = 2048 : i32} : memref<7168xbf16>
    aie.buffer @exact(%tile_0_2) : memref<512xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// Buffers are placed largest first. "large" needs a whole clean 16 kB bank.
// The stack occupies part of bank 0, so "large" takes bank 1 exactly. Placing
// the four small buffers first would spread them round-robin over every bank
// and dirty all four, which leaves "large" no bank-contained home.
// CHECK-LABEL: module @largest_first
// CHECK: aie.buffer @small0(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32} : memref<16xbf16>
// CHECK: aie.buffer @small1(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32} : memref<16xbf16>
// CHECK: aie.buffer @small2(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32} : memref<16xbf16>
// CHECK: aie.buffer @small3(%tile_0_2) {address = 32800 : i32, mem_bank = 2 : i32} : memref<16xbf16>
// CHECK: aie.buffer @large(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32} : memref<8192xbf16>
module @largest_first {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @small0(%tile_0_2) : memref<16xbf16>
    aie.buffer @small1(%tile_0_2) : memref<16xbf16>
    aie.buffer @small2(%tile_0_2) : memref<16xbf16>
    aie.buffer @small3(%tile_0_2) : memref<16xbf16>
    aie.buffer @large(%tile_0_2) : memref<8192xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// A zero-sized buffer is degenerate but legal, and must not upset the occupancy
// bookkeeping. It takes a free address where one exists.
// CHECK-LABEL: module @zero_sized_buffer
// CHECK: aie.buffer @empty(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32} : memref<0xi32>
// CHECK: aie.buffer @normal(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32} : memref<16xbf16>
module @zero_sized_buffer {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @empty(%tile_0_2) : memref<0xi32>
    aie.buffer @normal(%tile_0_2) : memref<16xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// Every bank is exactly full, so no hole exists anywhere. A zero-sized buffer
// covers no bytes, so allocation still places it, and the overlap checks accept
// the address it shares with the buffer or the stack.
// CHECK-LABEL: module @zero_sized_buffer_on_full_tile
// CHECK: aie.buffer @b0(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32} : memref<15360xi8>
// CHECK: aie.buffer @b1(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32} : memref<16384xi8>
// CHECK: aie.buffer @b2(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32} : memref<16384xi8>
// CHECK: aie.buffer @b3(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32} : memref<16384xi8>
// CHECK: aie.buffer @empty(%tile_0_2) {address = 0 : i32, mem_bank = 0 : i32} : memref<0xi32>
module @zero_sized_buffer_on_full_tile {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @b0(%tile_0_2) {address = 1024 : i32} : memref<15360xi8>
    aie.buffer @b1(%tile_0_2) {address = 16384 : i32} : memref<16384xi8>
    aie.buffer @b2(%tile_0_2) {address = 32768 : i32} : memref<16384xi8>
    aie.buffer @b3(%tile_0_2) {address = 49152 : i32} : memref<16384xi8>
    aie.buffer @empty(%tile_0_2) : memref<0xi32>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// The same, with the zero-sized buffer also pinning the bank it wants, and that
// bank exactly full. A requested mem_bank is a hard constraint, but a buffer
// that covers no bytes cannot exhaust a bank: the bank search fails only for
// want of a free byte the buffer never uses. It lands inside the bank it
// requested.
// CHECK-LABEL: module @zero_sized_buffer_pinned_to_full_bank
// CHECK: aie.buffer @b1(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32} : memref<16384xi8>
// CHECK: aie.buffer @empty(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32} : memref<0xi32>
module @zero_sized_buffer_pinned_to_full_bank {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @b1(%tile_0_2) {address = 16384 : i32} : memref<16384xi8>
    aie.buffer @empty(%tile_0_2) {mem_bank = 1 : i32} : memref<0xi32>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// A zero-sized buffer pinned by explicit `address` to the top of the tile's
// memory, one past every bank's range, is legal for the same reason: it covers
// no bytes, so nothing there conflicts with it. It lands in the last bank.
// CHECK-LABEL: module @zero_sized_buffer_at_exact_top_of_tile
// CHECK: aie.buffer @top(%tile_0_2) {address = 65536 : i32, mem_bank = 3 : i32} : memref<0xi32>
module @zero_sized_buffer_at_exact_top_of_tile {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    aie.buffer @top(%tile_0_2) {address = 65536 : i32} : memref<0xi32>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}
