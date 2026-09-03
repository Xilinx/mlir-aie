//===- test_mmap_data_region_record.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The "data" region comes from the buffer allocator's recorded placement
// (data_origin and data_length on aie.core).
//
// The layout below is one where the recorded placement and a fresh computation
// differ, so the test cannot pass by accident. Free space above the 0x400 stack
// is:
//     [0x400,  0x2000)  = 0x1C00 bytes
//     [0x4000, 0x8000)  = 0x4000 bytes   <-- what largestFreeRun picks
// The core records the *smaller* gap. A generator that recomputed the region
// would emit ORIGIN = 0x74000, LENGTH = 0x4000; the recorded placement gives
// ORIGIN = 0x70400, LENGTH = 0x1C00.
//
// test_mmap_data_region_gap.mlir covers the other half of this contract: IR
// with no recorded region, which never went through the allocator, gets a
// region computed here.

// RUN: aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s | FileCheck --check-prefix=LD02 %s

// LD02: MEMORY
// LD02-NEXT: {
// LD02-NEXT:    program (RX) : ORIGIN = 0, LENGTH = 0x4000
// LD02-NEXT:    data (!RX) : ORIGIN = 0x70400, LENGTH = 0x1C00
// LD02-NEXT: }

module @test_mmap_data_region_record {
 aie.device(npu1_1col) {
  %t02 = aie.tile(0, 2)

  // a: local [0x2000, 0x4000)
  %buf_a = aie.buffer(%t02) { sym_name = "a", address = 0x2000 : i32 } : memref<2048xi32>
  // gap: local [0x4000, 0x8000) -- larger, and deliberately NOT the one taken
  // b: local [0x8000, 0xC000)
  %buf_b = aie.buffer(%t02) { sym_name = "b", address = 0x8000 : i32 } : memref<4096xi32>
  // c: local [0xC000, 0x10000)
  %buf_c = aie.buffer(%t02) { sym_name = "c", address = 0xC000 : i32 } : memref<4096xi32>

  aie.core(%t02) {
    aie.end
  } {stack_size = 1024 : i32, data_origin = 1024 : i32, data_length = 7168 : i32}
 }
}
