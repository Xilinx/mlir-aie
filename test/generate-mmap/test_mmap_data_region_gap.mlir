//===- test_mmap_data_region_gap.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for the "data" region of the generated linker script.
//
// The "data" region is where the linker places compiler-generated sections
// (.data/.rodata/.bss) that are not explicitly placed buffers -- e.g. a
// kernel's static locals. Buffer addresses here are hand-specified to
// reproduce the fragmented-free-space layout that the bank-aware buffer
// allocator can produce on AIE2: a buffer ("c") is placed flush against the
// top of the 0x10000-byte local memory, while a free gap remains lower down
// (between "a" and "b").
//
// Before the fix, the generator computed the data region as *only* the space
// above the highest buffer, i.e. LENGTH = getLocalMemorySize() - max. Because
// "c" ends exactly at the top of memory (local 0x10000), that yielded
//     data (!RX) : ORIGIN = 0x80000, LENGTH = 0x0
// and any non-empty .bss/.data overflowed with
//     ld.lld: error: section '.bss' will not fit in region 'data'
// even though 0x4000 bytes were free in the [0x4000, 0x8000) gap.
//
// After the fix, the generator picks the largest contiguous free gap across
// the stack and this tile's buffers, so the data region is placed in that gap:
//     data (!RX) : ORIGIN = 0x74000, LENGTH = 0x4000   (0x70000 + 0x4000)

// RUN: aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s | FileCheck --check-prefix=LD02 %s

// LD02: MEMORY
// LD02-NEXT: {
// LD02-NEXT:    program (RX) : ORIGIN = 0, LENGTH = 0x0020000
// LD02-NEXT:    data (!RX) : ORIGIN = 0x74000, LENGTH = 0x4000
// LD02-NEXT: }

module @test_mmap_data_region_gap {
 aie.device(npu1_1col) {
  %t02 = aie.tile(0, 2)

  // a: local [0x2000, 0x4000)
  %buf_a = aie.buffer(%t02) { sym_name = "a", address = 0x2000 : i32 } : memref<2048xi32>
  // gap: local [0x4000, 0x8000) -- the largest free gap (0x4000 bytes)
  // b: local [0x8000, 0xC000)
  %buf_b = aie.buffer(%t02) { sym_name = "b", address = 0x8000 : i32 } : memref<4096xi32>
  // c: local [0xC000, 0x10000) -- fills memory flush to the top
  %buf_c = aie.buffer(%t02) { sym_name = "c", address = 0xC000 : i32 } : memref<4096xi32>

  aie.core(%t02) {
    aie.end
  }
 }
}
