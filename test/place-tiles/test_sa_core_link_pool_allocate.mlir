//===- test_sa_core_link_pool_allocate.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t 2> %t.err
// RUN: FileCheck %s --check-prefix=WARN --input-file=%t.err
// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses %t 2>&1 >/dev/null | FileCheck %s --check-prefix=ADDR

// A linked fifo with an aie.objectfifo.allocate shares memory, but the split
// still builds the link's one pool on the link tile and nothing else for the
// fifo, so the SA charges that pool there and no shared-memory pool.

// The output, with initial values, owns the 4 x 4 KB pool; the delegate is the
// link tile.
// WARN: [[@LINE+3]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: 'aie.buffer' op could not be placed: buffer "out__buf3"
module @delegate_on_link_tile {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<1024xi32>> = [dense<1> : memref<1024xi32>, dense<2> : memref<1024xi32>, dense<3> : memref<1024xi32>, dense<4> : memref<1024xi32>]
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%l)
    %big = aie.buffer(%l) {sym_name = "big"} : memref<13000xi32>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}

// -----

// The delegate is the producer's tile, but the pool still goes to the link
// tile.
// WARN: [[@LINE+5]]:3: warning: SA placer's resource estimate is over capacity
// WARN-NOT: over capacity
// ADDR: error: 'aie.buffer' op could not be placed: buffer "out__buf3"
// ADDR-NOT: error
module @delegate_on_producer_tile {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %l = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 4 : i32) : !aie.objectfifo<memref<1024xi32>> = [dense<1> : memref<1024xi32>, dense<2> : memref<1024xi32>, dense<3> : memref<1024xi32>, dense<4> : memref<1024xi32>]
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%a)
    %big = aie.buffer(%l) {sym_name = "big"} : memref<12500xi32>
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
  }
}
