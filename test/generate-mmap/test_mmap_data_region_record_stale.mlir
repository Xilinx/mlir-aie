//===- test_mmap_data_region_record_stale.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A recorded data region that collides with a buffer means the allocator's
// placement is stale: a pass added or moved a buffer on this tile after
// --aie-assign-buffer-addresses ran. Emitting the script would grant the core
// compiler a region that overlaps a buffer, and the two would corrupt each
// other at runtime.
//
// Recomputing the region here would hide that pipeline-ordering bug, so this is
// a hard error.

// RUN: not aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s 2>&1 | FileCheck %s

// CHECK: error: {{.*}}recorded data region 0x8000-0x8FFF overlaps buffer 'b' at 0x8000
// CHECK-SAME: placement is stale

module @test_mmap_data_region_record_stale {
 aie.device(npu1_1col) {
  %t02 = aie.tile(0, 2)

  %buf_a = aie.buffer(%t02) { sym_name = "a", address = 0x2000 : i32 } : memref<2048xi32>
  // b sits at 0x8000, exactly where the recorded region below claims to start.
  %buf_b = aie.buffer(%t02) { sym_name = "b", address = 0x8000 : i32 } : memref<4096xi32>

  aie.core(%t02) {
    aie.end
  } {stack_size = 1024 : i32, data_origin = 32768 : i32, data_length = 4096 : i32}
 }
}
