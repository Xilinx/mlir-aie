//===- alloc_group_alignment_padding.mlir -----------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// A unit's size must reflect the alignment padding placement actually inserts
// between a group's members, not just their summed sizes. Group "a" holds
// "a2" (80 B, needs 64B vector alignment) then "a1" (64 B, also needs 64B
// alignment, placed second because it is smaller): a1 lands at 128, not 80,
// since 80 is not 64-aligned. So group "a" really spans 192 B while the naive
// sum of its members is only 144 B. Group "b" ("b1", 16 B) is smaller either
// way and doesn't change which group drives the unit's size.
//
// The unit must advance the cursor to 192 (the real end), not 144 (the naive
// sum): with the bug, "other" lands at 160, inside a1's [128,192) range, and
// the allocator reports a false overlap instead of succeeding.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s
// CHECK: aie.buffer({{.*}}) {address = 0 : i32, alloc_group = "a", sym_name = "a2"} : memref<20xi32>
// CHECK: aie.buffer({{.*}}) {address = 128 : i32, alloc_group = "a", sym_name = "a1"} : memref<16xi32>
// CHECK: aie.buffer({{.*}}) {address = 0 : i32, alloc_group = "b", sym_name = "b1"} : memref<4xi32>
// CHECK: aie.buffer({{.*}}) {address = 192 : i32, sym_name = "other"} : memref<2xi32>
module @test_alloc_group_alignment_padding {
  aie.device(npu2) {
    %0 = aie.tile(0, 2)
    %a2 = aie.buffer(%0) { sym_name = "a2", alloc_group = "a" } : memref<20xi32>
    %a1 = aie.buffer(%0) { sym_name = "a1", alloc_group = "a" } : memref<16xi32>
    %b1 = aie.buffer(%0) { sym_name = "b1", alloc_group = "b" } : memref<4xi32>
    %other = aie.buffer(%0) { sym_name = "other" } : memref<2xi32>
  }
}
