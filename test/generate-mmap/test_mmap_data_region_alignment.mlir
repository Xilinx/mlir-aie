//===- test_mmap_data_region_alignment.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The emitted "data" ORIGIN is aligned. The linker starts the core's .data at a
// multiple of its strongest section alignment, so an ORIGIN at an arbitrary
// byte costs the region that much padding, which is enough to overflow a region
// of the exact size.
//
// This is the no-stamp path (see test_mmap_data_region_gap.mlir): the region is
// derived here, so the alignment is applied here and matches the value the
// allocator stamps.
//
// The 3-byte buffer at 0x400 ends at 0x403. npu1 requires 256-bit (32-byte)
// alignment, so the region starts at 0x420.

// RUN: aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s | FileCheck --check-prefix=LD02 %s

// LD02: data (!RX) : ORIGIN = 0x70420, LENGTH = 0xFBE0

module @test_mmap_data_region_alignment {
 aie.device(npu1_1col) {
  %t02 = aie.tile(0, 2)

  %buf_odd = aie.buffer(%t02) { sym_name = "odd", address = 0x400 : i32 } : memref<3xi8>

  aie.core(%t02) {
    aie.end
  } {stack_size = 1024 : i32}
 }
}
