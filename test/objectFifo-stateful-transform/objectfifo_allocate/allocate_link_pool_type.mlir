//===- allocate_link_pool_type.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %s

// A linked fifo with an aie.objectfifo.allocate shares memory: its core fills
// the link's one pool directly, one pool object per acquire. When the pool's
// owner is the larger output, those objects are not the fifo's, so the
// allocate is rejected. Before, the stateful transform failed later with
// "acquired object type 'memref<1024xi32>' does not match".

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
