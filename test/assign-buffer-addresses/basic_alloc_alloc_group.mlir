//===- basic_alloc_alloc_group.mlir ----------------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// Buffers in DIFFERENT alloc_groups are overlaid: the two groups share one
// region sized at the larger group's total, so the next ungrouped buffer starts
// right after it. Here "big" is 2048 B in group "a" and "small" is 256 B in
// group "b"; both land at 1024, and "other" lands at 1024 + 2048 = 3072.
// Without the groups it would be at 3328.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, alloc_group = "a", sym_name = "big"} : memref<512xi32>
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, alloc_group = "b", sym_name = "small"} : memref<64xi32>
// CHECK: aie.buffer({{.*}}) {address = 3072 : i32, sym_name = "other"} : memref<128xi32>
module @test_alloc_group {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %big = aie.buffer(%0) { sym_name = "big", alloc_group = "a" } : memref<512xi32>
    %small = aie.buffer(%0) { sym_name = "small", alloc_group = "b" } : memref<64xi32>
    %other = aie.buffer(%0) { sym_name = "other" } : memref<128xi32>
    aie.core(%0) {
      aie.end
    }
  }
}
