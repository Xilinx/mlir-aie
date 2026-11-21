//===- allocate_link_pool_type.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s

// A linked fifo with an aie.objectfifo.allocate whose ends reach its delegate
// shares memory: its core fills the link's one pool directly, one pool object
// per acquire. When the pool's owner is the larger output, those objects are
// not the fifo's, so the allocate is rejected once the tiles are placed.

module @larger_output {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    // expected-error@+1 {{places @in0's objects in the link's pool, whose objects are @out's 'memref<2048xi32>'; a fifo that shares memory with a link must have the pool's object type (@in0's is 'memref<1024xi32>')}}
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

module @larger_output_placed {
  aie.device(npu2_1col) {
    %s = aie.tile(0, 0)
    %a = aie.tile(0, 2)
    %l = aie.tile(0, 3)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<2048xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    // expected-error@+1 {{places @in0's objects in the link's pool, whose objects are @out's 'memref<2048xi32>'}}
    aie.objectfifo.allocate @in0 (%l)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// The output owns the pool for its initial values, with the input's object
// type, so the allocate stands, with the delegate on the link tile or on the
// producer's.
module @init_values_on_link_tile {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<1024xi32>> = [dense<1> : memref<1024xi32>, dense<2> : memref<1024xi32>, dense<3> : memref<1024xi32>, dense<4> : memref<1024xi32>]
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    %big = aie.buffer(%l) {sym_name = "big"} : memref<9500xi32>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

module @init_values_on_producer_tile {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<1024xi32>> = [dense<1> : memref<1024xi32>, dense<2> : memref<1024xi32>, dense<3> : memref<1024xi32>, dense<4> : memref<1024xi32>]
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%a)
    %big = aie.buffer(%l) {sym_name = "big"} : memref<100xi32>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// A join's input whose segment holds two of its objects would fill it in two
// acquires, so it cannot be the pool's there.
module @join_segment_of_two_objects {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %f = aie.tile(0, 5)
    aie.objectfifo @in0(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @in1(%f, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 16] [])
    // expected-error@+1 {{a fifo that shares memory with a link must have the pool's object type (@in0's is 'memref<8xi32>'), or as a join's input or a distribute's output, its segment's}}
    aie.objectfifo.allocate @in0 (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<8xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}

// -----

// A two-dimensional lane leading a join is a run of the pool's elements only
// as a rank-1 view, which its acquires do not name.
module @join_two_dimensional_lane {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %f = aie.tile(0, 5)
    aie.objectfifo @in0(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<4x16xi32>>
    aie.objectfifo @in1(%f, {%t}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @out(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo.link [@in0, @in1] -> [@out] ([0, 64] [])
    // expected-error@+1 {{a fifo that shares memory with a link must have the pool's object type (@in0's is 'memref<4x16xi32>'), or as a join's input or a distribute's output, its segment's}}
    aie.objectfifo.allocate @in0 (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<4x16xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @in1(Produce, 1) : memref<64xi32>
      aie.objectfifo.release @in1(Produce, 1)
      aie.end
    }
  }
}

// -----

// A distribute whose two-dimensional pool object leads with a lane's run has
// no rank-1 view of it to give that lane's core.
module @distribute_two_dimensional_pool {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %f = aie.tile(0, 5)
    aie.objectfifo @in(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<2x16xi32>>
    aie.objectfifo @out0(%t, {%t}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out1(%t, {%f}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in] -> [@out0, @out1] ([] [0, 16])
    aie.objectfifo.allocate @in (%t)
    // expected-error@+1 {{a fifo that shares memory with a link must have the pool's object type (@out0's is 'memref<16xi32>'), or as a join's input or a distribute's output, its segment's}}
    aie.objectfifo.allocate @out0 (%t)
    aie.core(%t) {
      %0 = aie.objectfifo.acquire @in(Produce, 1) : memref<2x16xi32>
      aie.objectfifo.release @in(Produce, 1)
      %1 = aie.objectfifo.acquire @out0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out0(Consume, 1)
      aie.end
    }
    aie.core(%f) {
      %0 = aie.objectfifo.acquire @out1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @out1(Consume, 1)
      aie.end
    }
  }
}
