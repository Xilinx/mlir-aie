//===- basic_alloc_alloc_group_error.mlir ----------------------*- MLIR -*-===//
//
//
//===----------------------------------------------------------------------===//

// Two buffers from DIFFERENT alloc_groups referenced by the SAME core with no
// selector between them are simultaneously live by construction, so overlaying
// them would alias live data. That half of the contract is decidable, and is
// rejected.

// RUN: not aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s 2>&1 | FileCheck %s
// CHECK: error: {{.*}}is in alloc_group 'b' while this aie.core also references a buffer in alloc_group 'a'
module @test_alloc_group_same_core {
  aie.device(xcvc1902) {
    %0 = aie.tile(3, 3)
    %big = aie.buffer(%0) { sym_name = "big", alloc_group = "a" } : memref<512xi32>
    %small = aie.buffer(%0) { sym_name = "small", alloc_group = "b" } : memref<64xi32>
    aie.core(%0) {
      %c0 = arith.constant 0 : index
      %v = memref.load %big[%c0] : memref<512xi32>
      %w = memref.load %small[%c0] : memref<64xi32>
      aie.end
    }
  }
}
