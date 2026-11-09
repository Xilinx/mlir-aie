//===- test_sa_allocate_home_unreachable.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s 2>&1 | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=2" %s 2>&1 | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=3" %s 2>&1 | FileCheck %s

// The split keeps an allocated fifo's objects at
// ObjectFifoCreateOp::allocatedHome, which both ends must reach, pinned
// aie.tile ends included: a linked fifo's at the link's pool on the link tile.
// No placement of these designs gives in0 or @f a home its ends reach, so the
// SA fails on its delegate penalty. Before, it checked only the logical ends'
// reach of the delegate and stayed silent, and the split or addressing failed.

// CHECK: [[@LINE+4]]:3: error: SA placer failed to find a legal placement (cascade and shared memory penalty=0, delegate penalty={{[1-9][0-9]*}})
// A on (0,2) reaches only (0,3)'s memory, which another core holds; the link
// tile reaches A's memory from (1,2), but A cannot reach the pool there.
module @link_tile_east {
  aie.device(npu2_2col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.tile(0, 2)
    %x3 = aie.tile(0, 3)
    %x4 = aie.tile(0, 4)
    %x5 = aie.tile(0, 5)
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%a)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%x3) { aie.end }
    aie.core(%x4) { aie.end }
    aie.core(%x5) { aie.end }
  }
}

// -----

// CHECK: [[@LINE+4]]:3: error: SA placer failed to find a legal placement (cascade and shared memory penalty=0, delegate penalty={{[1-9][0-9]*}})
// The delegate (0,3) is reached by A and by a link tile on (0,4), but A
// cannot reach the link's pool there.
module @delegate_between {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %a = aie.tile(0, 2)
    %d = aie.tile(0, 3)
    %l = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @in0(%a, {%l}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%l, {%s}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.link [@in0] -> [@out] ([] [])
    aie.objectfifo.allocate @in0 (%d)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @in0(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @in0(Produce, 1)
      aie.end
    }
    aie.core(%d) { aie.end }
  }
}

// -----

// CHECK: [[@LINE+4]]:3: error: SA placer failed to find a legal placement (cascade and shared memory penalty=0, delegate penalty={{[1-9][0-9]*}})
// The pinned producer on (0,2) cannot reach the delegate on (0,5), wherever B
// goes.
module @pinned_end {
  aie.device(npu2_1col) {
    %a = aie.tile(0, 2)
    %d = aie.tile(0, 5)
    %b = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @f(%a, {%b}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo.allocate @f (%d)
    aie.core(%a) {
      %0 = aie.objectfifo.acquire @f(Produce, 1) : memref<1024xi32>
      aie.objectfifo.release @f(Produce, 1)
      aie.end
    }
    aie.core(%b) {
      %0 = aie.objectfifo.acquire @f(Consume, 1) : memref<1024xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
  }
}
