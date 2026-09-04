//===- bad_data_region.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// data_origin and data_length record where the buffer allocator placed the
// core's data region. The allocator writes them, so these cases check that the
// verifier rejects a malformed pair, before it becomes a linker script that
// overlaps a buffer or extends past the end of the tile.

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

// CHECK: error{{.*}}'data_origin' and 'data_length' must be set together
module @origin_without_length {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_origin = 4096 : i32}
  }
}

// -----

// CHECK: error{{.*}}'data_origin' and 'data_length' must be set together
module @length_without_origin {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_length = 4096 : i32}
  }
}

// -----

// CHECK: error{{.*}}op data region at 0x200 starts below the stack (1024 bytes)
module @starts_inside_stack {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_origin = 512 : i32, data_length = 4096 : i32}
  }
}

// -----

// CHECK: error{{.*}}op data region 0xF000-0x10FFF runs past the end of this tile's memory (65536 bytes total)
module @runs_past_end {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_origin = 61440 : i32, data_length = 8192 : i32}
  }
}

// -----

// The linker starts .data at a multiple of its strongest section alignment, so
// an unaligned origin costs the region that much. Both producers align it, so
// an unaligned value is hand-written or stale.
// CHECK: error{{.*}}op data region at 0x403 is not aligned to 64 bytes
module @unaligned_origin {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, data_origin = 1027 : i32, data_length = 4096 : i32}
  }
}

// -----

// The grant satisfies the request, so a grant smaller than the request is a
// bookkeeping bug in whatever wrote it.
// CHECK: error{{.*}}op granted data region is 4096 bytes, smaller than the requested reserved_data_size of 8192 bytes
module @grant_smaller_than_request {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %core = aie.core(%t) { aie.end } {stack_size = 1024 : i32, reserved_data_size = 8192 : i32, data_origin = 4096 : i32, data_length = 4096 : i32}
  }
}
