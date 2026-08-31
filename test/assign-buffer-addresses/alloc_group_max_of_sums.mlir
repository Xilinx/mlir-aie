//===- alloc_group_max_of_sums.mlir ----------------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// A group is a MODE, so N modes cost max(sum of each mode), not sum(max of each
// pairing). Mode "a" holds 2048 B + 256 B, mode "b" holds 256 B + 2048 B. Both
// total 2304 B, so the shared region is 2304 B and "other" follows at
// 1024 + 2304 = 3328.
//
// Pairing the buffers across modes instead -- the only thing a mutual-exclusion
// attribute could express -- would give max(2048,256) + max(256,2048) = 4096 B
// and put "other" at 5120. It would also require the two modes to hold the same
// number of buffers, which grouping by mode does not.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s
// Within a group members follow the pass's existing largest-first order, so each
// group places its 2048 B member at the base and its 256 B member above it. The
// two groups overlay, which is why a2 and b1 share 3072.

// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, alloc_group = "a", sym_name = "a1"} : memref<512xi32>
// CHECK: aie.buffer({{.*}}) {address = 3072 : i32, alloc_group = "a", sym_name = "a2"} : memref<64xi32>
// CHECK: aie.buffer({{.*}}) {address = 3072 : i32, alloc_group = "b", sym_name = "b1"} : memref<64xi32>
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, alloc_group = "b", sym_name = "b2"} : memref<512xi32>
// CHECK: aie.buffer({{.*}}) {address = 3328 : i32, sym_name = "other"} : memref<128xi32>
module @test_alloc_group_max_of_sums {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %a1 = aie.buffer(%0) { sym_name = "a1", alloc_group = "a" } : memref<512xi32>
    %a2 = aie.buffer(%0) { sym_name = "a2", alloc_group = "a" } : memref<64xi32>
    %b1 = aie.buffer(%0) { sym_name = "b1", alloc_group = "b" } : memref<64xi32>
    %b2 = aie.buffer(%0) { sym_name = "b2", alloc_group = "b" } : memref<512xi32>
    %other = aie.buffer(%0) { sym_name = "other" } : memref<128xi32>
    %rtp = aie.buffer(%0) { sym_name = "rtp" } : memref<1xi32>
    aie.core(%0) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %sel = memref.load %rtp[%c0] : memref<1xi32>
      %is_a = arith.cmpi eq, %sel, %c0_i32 : i32
      scf.if %is_a {
        %v = memref.load %a1[%c0] : memref<512xi32>
        %u = memref.load %a2[%c0] : memref<64xi32>
      } else {
        %w = memref.load %b1[%c0] : memref<64xi32>
        %x = memref.load %b2[%c0] : memref<512xi32>
      }
      aie.end
    }
  }
}
