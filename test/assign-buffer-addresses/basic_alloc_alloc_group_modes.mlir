//===- basic_alloc_alloc_group_modes.mlir ----------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// The case alloc_group exists to serve: one core carrying two mode bodies, with
// a runtime selector picking one per dispatch. The two buffers are in different
// groups and are referenced from different branches of one scf.if, so they are
// never live together and the overlay is accepted -- both land at 1024 and
// "other" follows the larger.
//
// Contrast basic_alloc_alloc_group_error.mlir, where nothing separates the two
// references and the same annotation is rejected.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, alloc_group = "a", sym_name = "big"} : memref<512xi32>
// CHECK: aie.buffer({{.*}}) {address = 1024 : i32, alloc_group = "b", sym_name = "small"} : memref<64xi32>
// CHECK: aie.buffer({{.*}}) {address = 3072 : i32, sym_name = "other"} : memref<128xi32>
module @test_alloc_group_modes {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %big = aie.buffer(%0) { sym_name = "big", alloc_group = "a" } : memref<512xi32>
    %small = aie.buffer(%0) { sym_name = "small", alloc_group = "b" } : memref<64xi32>
    %other = aie.buffer(%0) { sym_name = "other" } : memref<128xi32>
    %rtp = aie.buffer(%0) { sym_name = "rtp" } : memref<1xi32>
    aie.core(%0) {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %sel = memref.load %rtp[%c0] : memref<1xi32>
      %is_a = arith.cmpi eq, %sel, %c0_i32 : i32
      scf.if %is_a {
        %v = memref.load %big[%c0] : memref<512xi32>
        memref.store %v, %other[%c0] : memref<128xi32>
      } else {
        %w = memref.load %small[%c0] : memref<64xi32>
        memref.store %w, %other[%c0] : memref<128xi32>
      }
      aie.end
    }
  }
}
