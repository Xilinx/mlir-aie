//===- bank_aware_alloc_over_bank.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A buffer larger than one bank used to be unplaceable by the bank-aware
// scheme no matter how much memory was free, because placement was required to
// fit inside a single bank. Spreading buffers over banks is a way to limit DMA
// bank contention, not a bound on how large a buffer may be, so a buffer that
// fits in no single bank straddles bank boundaries instead of failing.

// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// A 512 kB memtile as 8 banks of 64 kB, completely empty. A 128 kB buffer used
// to be rejected here with ~384 kB free.
// CHECK-LABEL: module @over_bank_empty_memtile
// CHECK: %big = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "big"} : memref<32768xi32>
module @over_bank_empty_memtile {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %big = aie.buffer(%mem_tile_0_1) {sym_name = "big"} : memref<32768xi32>
    aie.memtile_dma(%mem_tile_0_1) {
      aie.end
    }
  }
}

// -----

// With bank 0 pinned full, the 128 kB buffer starts at the next bank boundary.
// A bank-aligned start is preferred because for a given size it is the
// placement touching the fewest banks: 65536 spans banks 1-2, whereas any
// unaligned start of a 128 kB buffer would touch three banks.
// CHECK-LABEL: module @over_bank_after_obstacle
// CHECK: %big = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "big"} : memref<32768xi32>
module @over_bank_after_obstacle {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %pin_bank0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, sym_name = "pin_bank0"} : memref<16384xi32>
    %big = aie.buffer(%mem_tile_0_1) {sym_name = "big"} : memref<32768xi32>
    aie.memtile_dma(%mem_tile_0_1) {
      aie.end
    }
  }
}

// -----

// Straddling banks is a last resort, not a preference: a buffer that does fit
// inside a single bank still gets one, and the round-robin spread over banks
// is unchanged.
// CHECK-LABEL: module @fits_in_bank_still_spreads
// CHECK: %a = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<4096xi32>
// CHECK: %b = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<4096xi32>
// CHECK: %c = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "c"} : memref<4096xi32>
module @fits_in_bank_still_spreads {
  aie.device(npu2) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %a = aie.buffer(%mem_tile_0_1) {sym_name = "a"} : memref<4096xi32>
    %b = aie.buffer(%mem_tile_0_1) {sym_name = "b"} : memref<4096xi32>
    %c = aie.buffer(%mem_tile_0_1) {sym_name = "c"} : memref<4096xi32>
    aie.memtile_dma(%mem_tile_0_1) {
      aie.end
    }
  }
}

// -----

// Straddling must not run over the stack. On a core tile the 32 kB buffer
// needs two of the four 16 kB banks; bank 0 is unusable because the stack sits
// at its base, so the buffer starts at the bank 1 boundary.
// CHECK-LABEL: module @over_bank_skips_stack
// CHECK: %big = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "big"} : memref<16384xbf16>
module @over_bank_skips_stack {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %big = aie.buffer(%tile_0_2) {sym_name = "big"} : memref<16384xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}

// -----

// A pinned buffer may itself straddle a bank boundary. The hardware places no
// natural-size alignment requirement on a buffer, and designs in the field
// hand-pin buffers at addresses like this; the allocator used to report
// "allocated buffers exceeded available memory" for the tile.
// CHECK-LABEL: module @pin_straddles_bank_boundary
// CHECK: %straddle = aie.buffer(%tile_0_2) {address = 8192 : i32, mem_bank = 0 : i32, sym_name = "straddle"} : memref<8192xbf16>
// CHECK: %after = aie.buffer(%tile_0_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "after"} : memref<512xbf16>
module @pin_straddles_bank_boundary {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %straddle = aie.buffer(%tile_0_2) {address = 8192 : i32, sym_name = "straddle"} : memref<8192xbf16>
    %after = aie.buffer(%tile_0_2) {mem_bank = 1 : i32, sym_name = "after"} : memref<512xbf16>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32}
  }
}
