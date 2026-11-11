//===- broadcast_to_producer_tile.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectfifo-split %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectfifo-split %s | FileCheck %s
// RUN: aie-opt --split-input-file --aie-place-tiles --aie-objectFifo-stateful-transform %s | FileCheck %s --check-prefix=STATEFUL

// @f goes from (7, 4) to a core and back to (7, 4), whose core acquires 3
// objects to produce and 1 to consume. Each end's depth counts only its own
// core's acquires: 4 objects to produce and 2 to consume on (7, 4), 3 on the
// other core, which acquires 2. Before, the consuming end on (7, 4) counted
// the producing acquires too, the placers planned it 4 deep, the split made
// it 2 deep, and the split's check of that plan asserted.
// CHECK-LABEL: module @physical
// CHECK: aie.objectfifo.pool @f__pool(%{{.*}}) {depth = 4 : i32
// CHECK: aie.objectfifo.pool @f__cons0__pool(%{{.*}}) {depth = 3 : i32
// CHECK: aie.objectfifo.pool @f__cons1__pool(%{{.*}}) {depth = 2 : i32
module @physical {
  aie.device(xcvc1902) {
    %p = aie.tile(7, 4)
    %c = aie.tile(7, 2)
    aie.objectfifo @f(%p, {%c, %p}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %a0, %a1, %a2 = aie.objectfifo.acquire @f(Produce, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @f(Produce, 3)
      %b = aie.objectfifo.acquire @f(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %d0, %d1 = aie.objectfifo.acquire @f(Consume, 2) : memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @f(Consume, 2)
      aie.end
    }
  }
}

// -----

// The same with the other core left to a placer, which plans each end as the
// split then makes it.
// CHECK-LABEL: module @logical
// CHECK: aie.objectfifo.pool @f__pool(%{{.*}}) {depth = 4 : i32
// CHECK: aie.objectfifo.pool @f__cons0__pool(%{{.*}}) {depth = 3 : i32
// CHECK: aie.objectfifo.pool @f__cons1__pool(%{{.*}}) {depth = 2 : i32
module @logical {
  aie.device(xcvc1902) {
    %p = aie.tile(7, 4)
    %c = aie.logical_tile<CoreTile>(?, ?)
    aie.objectfifo @f(%p, {%c, %p}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %a0, %a1, %a2 = aie.objectfifo.acquire @f(Produce, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @f(Produce, 3)
      %b = aie.objectfifo.acquire @f(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f(Consume, 1)
      aie.end
    }
    aie.core(%c) {
      %d0, %d1 = aie.objectfifo.acquire @f(Consume, 2) : memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @f(Consume, 2)
      aie.end
    }
  }
}

// -----

// A producer's core that acquires more to consume than to produce: its
// producing end holds only the 1 object it acquires plus 1 in flight, where
// the producing acquire and the consuming one used to share the larger depth.
// CHECK-LABEL: module @consume_larger
// CHECK: aie.objectfifo.pool @f__pool(%{{.*}}) {depth = 2 : i32
// CHECK: aie.objectfifo.pool @f__cons0__pool(%{{.*}}) {depth = 3 : i32
// CHECK: aie.objectfifo.pool @f__cons1__pool(%{{.*}}) {depth = 4 : i32
// STATEFUL-LABEL: module @consume_larger
// STATEFUL: %f__buf1 = aie.buffer
// STATEFUL-NOT: %f__buf2 = aie.buffer
module @consume_larger {
  aie.device(xcvc1902) {
    %p = aie.tile(7, 4)
    %c = aie.tile(7, 2)
    aie.objectfifo @f(%p, {%c, %p}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %a = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f(Produce, 1)
      %b0, %b1, %b2 = aie.objectfifo.acquire @f(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @f(Consume, 3)
      aie.end
    }
    aie.core(%c) {
      %d0, %d1 = aie.objectfifo.acquire @f(Consume, 2) : memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @f(Consume, 2)
      aie.end
    }
  }
}

// -----

// Not a broadcast: a fifo from a core back to its own tile through the
// tile's DMAs. The producing end is 2 deep (it was 4), the consuming end 4;
// the tile holds 6 buffers and 6 BDs, where it held 8.
// CHECK-LABEL: module @self_loop_dma
// CHECK: aie.objectfifo.pool @f__pool(%{{.*}}) {depth = 2 : i32
// CHECK: aie.objectfifo.pool @f__cons__pool(%{{.*}}) {depth = 4 : i32
// STATEFUL-LABEL: module @self_loop_dma
// STATEFUL: %f__buf1 = aie.buffer
// STATEFUL-NOT: %f__buf2 = aie.buffer
// STATEFUL: %f__cons__buf3 = aie.buffer
// STATEFUL-NOT: %f__cons__buf4 = aie.buffer
// STATEFUL: aie.mem(
// STATEFUL-COUNT-6: aie.dma_bd(
// STATEFUL-NOT: aie.dma_bd(
module @self_loop_dma {
  aie.device(npu2) {
    %p = aie.tile(1, 3)
    aie.objectfifo @f(%p, {%p}, 4 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
    aie.core(%p) {
      %a = aie.objectfifo.acquire @f(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f(Produce, 1)
      %b0, %b1, %b2 = aie.objectfifo.acquire @f(Consume, 3) : memref<16xi32>, memref<16xi32>, memref<16xi32>
      aie.objectfifo.release @f(Consume, 3)
      aie.end
    }
  }
}
