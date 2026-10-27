// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-assign-buffer-addresses %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Raw bytes fit exactly, but the generated buffer's aligned start does not.
module @alignment_spill {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    aie.buffer @fixed(%home) : memref<524285xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<3xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 3 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%home) drains @p
  }
}
// CHECK-LABEL: module @alignment_spill
// CHECK-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[NEXT:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer @p_buff_0(%[[NEXT]]) {address = 0 : i32{{.*}}}
// CHECK-DAG: aie.buffer @fixed(%[[HOME]]) {address = 0 : i32{{.*}}}

// -----

// Alignment applies to each object, not just to the start of the pool.
module @per_object_alignment {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    aie.buffer @fixed(%home) : memref<524281xi8>
    aie.objectfifo.pool @p(%home) {depth = 2 : i32} : memref<3xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 3 : i32}
    }
  }
}
// CHECK-LABEL: module @per_object_alignment
// CHECK-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[NEXT:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer @p_buff_0(%[[HOME]]) {address = 524284 : i32{{.*}}}
// CHECK-DAG: aie.buffer @p_buff_1(%[[NEXT]]) {address = 0 : i32{{.*}}}

// -----

// A fixed address splits free memory. Skip that extent before testing the
// generated buffer; the sum of bytes alone would incorrectly keep it local.
module @pinned_extent {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    aie.buffer @fixed(%home) {address = 4 : i32} : memref<524280xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<8xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 8 : i32}
    }
  }
}
// CHECK-LABEL: module @pinned_extent
// CHECK: %[[NEXT:.*]] = aie.tile(1, 1)
// CHECK: aie.buffer @p_buff_0(%[[NEXT]]) {address = 0 : i32{{.*}}}

// -----

// Preserve a usable hole below an address pin instead of counting the pin's
// end as occupied memory.
module @pinned_hole_exact_fit {
  aie.device(npu2_1col) {
    %home = aie.tile(0, 1)
    aie.buffer @fixed(%home) {address = 4 : i32} : memref<524284xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<4xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 4 : i32}
    }
  }
}
// CHECK-LABEL: module @pinned_hole_exact_fit
// CHECK: aie.buffer @p_buff_0({{.*}}) {address = 0 : i32{{.*}}}

// -----

// An unaligned final buffer needs no padding. The generated buffer is emitted
// before the equal-sized fixed buffer, so the aligned one is placed first.
module @unaligned_exact_fit {
  aie.device(npu2_1col) {
    %home = aie.tile(0, 1)
    aie.buffer @reserved(%home) : memref<524284xi8>
    aie.buffer @fixed(%home) {aligned = false} : memref<2xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<2xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 2 : i32}
    }
  }
}
// CHECK-LABEL: module @unaligned_exact_fit
// CHECK: aie.buffer @p_buff_0({{.*}}) {address = 524284 : i32{{.*}}}
// CHECK: aie.buffer @fixed({{.*}}) {address = 524286 : i32, aligned = false{{.*}}}
