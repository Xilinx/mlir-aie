// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-assign-buffer-addresses %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-assign-buffer-addresses %s -o /dev/null

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Raw bytes fit exactly, but the generated buffer's aligned start does not.
module @alignment_spill {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %fixed = aie.buffer(%home) {sym_name = "fixed"} : memref<524285xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<3xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 3 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%home) drains @p
  }
}
// CHECK-LABEL: module @alignment_spill
// CHECK-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[NEXT:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[NEXT]]) {address = 0 : i32, {{.*}}sym_name = "p_buff_0"}
// CHECK-DAG: aie.buffer(%[[HOME]]) {address = 0 : i32, {{.*}}sym_name = "fixed"}

// -----

// Alignment applies to each object, not just to the start of the pool.
module @per_object_alignment {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %fixed = aie.buffer(%home) {sym_name = "fixed"} : memref<524281xi8>
    aie.objectfifo.pool @p(%home) {depth = 2 : i32} : memref<3xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 3 : i32}
    }
  }
}
// CHECK-LABEL: module @per_object_alignment
// CHECK-DAG: %[[HOME:.*]] = aie.tile(0, 1)
// CHECK-DAG: %[[NEXT:.*]] = aie.tile(1, 1)
// CHECK-DAG: aie.buffer(%[[HOME]]) {address = 524284 : i32, {{.*}}sym_name = "p_buff_0"}
// CHECK-DAG: aie.buffer(%[[NEXT]]) {address = 0 : i32, {{.*}}sym_name = "p_buff_1"}

// -----

// A fixed address splits free memory. Skip that extent before testing the
// generated buffer; the sum of bytes alone would incorrectly keep it local.
module @pinned_extent {
  aie.device(npu2) {
    %home = aie.tile(0, 1)
    %fixed = aie.buffer(%home) {sym_name = "fixed", address = 4 : i32} : memref<524280xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<8xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 8 : i32}
    }
  }
}
// CHECK-LABEL: module @pinned_extent
// CHECK: %[[NEXT:.*]] = aie.tile(1, 1)
// CHECK: aie.buffer(%[[NEXT]]) {address = 0 : i32, {{.*}}sym_name = "p_buff_0"}

// -----

// Preserve a usable hole below an address pin instead of counting the pin's
// end as occupied memory.
module @pinned_hole_exact_fit {
  aie.device(npu2_1col) {
    %home = aie.tile(0, 1)
    %fixed = aie.buffer(%home) {sym_name = "fixed", address = 4 : i32} : memref<524284xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<4xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 4 : i32}
    }
  }
}
// CHECK-LABEL: module @pinned_hole_exact_fit
// CHECK: aie.buffer({{.*}}) {address = 0 : i32, {{.*}}sym_name = "p_buff_0"}

// -----

// An unaligned final buffer needs no padding. The generated buffer is emitted
// before the equal-sized fixed buffer, so the aligned one is placed first.
module @unaligned_exact_fit {
  aie.device(npu2_1col) {
    %home = aie.tile(0, 1)
    %reserved = aie.buffer(%home) {sym_name = "reserved"} : memref<524284xi8>
    %fixed = aie.buffer(%home) {sym_name = "fixed", aligned = false} : memref<2xi8>
    aie.objectfifo.pool @p(%home) {depth = 1 : i32} : memref<2xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 2 : i32}
    }
  }
}
// CHECK-LABEL: module @unaligned_exact_fit
// CHECK: aie.buffer({{.*}}) {address = 524284 : i32, {{.*}}sym_name = "p_buff_0"}
// CHECK: aie.buffer({{.*}}) {address = 524286 : i32, aligned = false, {{.*}}sym_name = "fixed"}
