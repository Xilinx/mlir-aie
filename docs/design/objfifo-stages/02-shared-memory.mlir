//===- 02-shared-memory.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESIGN SKETCH -- not a lit test. See 00-model.mlir first.
//
// Memory-adjacent tiles: one pool, two cores, no DMAs and no flow. Derived from
// test/objectFifo-stateful-transform/base/base_test_AIE2.mlir (@of0), where
// tile(1,2) and tile(1,3) are adjacent and @of0 lowers to a single set of
// of0_buff_* and locks on tile(1,2).
//
// The same-core fifo is the same picture with both cores on one tile; it is at
// the bottom of this file.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STAGE 0 -- input
//===----------------------------------------------------------------------===//

module @stage0 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

//===----------------------------------------------------------------------===//
// STAGE 1 -- after --aie-objectfifo-split
//
// ONE pool, on the tile whose memory holds the buffers, with a core endpoint at
// each end. The draining endpoint sits on tile13 and reaches tile12's memory,
// which is what makes this the shared-memory case.
//===----------------------------------------------------------------------===//

module @stage1 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo.pool @of0_pool(%tile12) {
      depth = 4 : i32, segments = [<offset = 0, size = 16>]
    } : memref<16xi32>

    aie.objectfifo.core_endpoint @of0_prod(%tile12) fills  @of0_pool
    aie.objectfifo.core_endpoint @of0_cons(%tile13) drains @of0_pool
  }
}

//===----------------------------------------------------------------------===//
// STAGE 2 -- after --aie-objectfifo-allocate
//
// Four buffers and one lock pair, on the pool's tile. No channels are assigned
// and no aie.flow is emitted, there being no DMA endpoint.
//===----------------------------------------------------------------------===//

module @stage2 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    %b0 = aie.buffer(%tile12) {sym_name = "of0_buff_0"} : memref<16xi32>
    %b1 = aie.buffer(%tile12) {sym_name = "of0_buff_1"} : memref<16xi32>
    %b2 = aie.buffer(%tile12) {sym_name = "of0_buff_2"} : memref<16xi32>
    %b3 = aie.buffer(%tile12) {sym_name = "of0_buff_3"} : memref<16xi32>
    %pl = aie.lock(%tile12) {init = 4 : i32, sym_name = "of0_prod_lock_0"}
    %cl = aie.lock(%tile12) {init = 0 : i32, sym_name = "of0_cons_lock_0"}

    aie.objectfifo.pool @of0_pool(%tile12) {
      depth    = 4 : i32,
      buffers  = [@of0_buff_0, @of0_buff_1, @of0_buff_2, @of0_buff_3],
      segments = [<produceLock = @of0_prod_lock_0, consumeLock = @of0_cons_lock_0,
                   offset = 0, size = 16>]
    } : memref<16xi32>

    aie.objectfifo.core_endpoint @of0_prod(%tile12) fills  @of0_pool
    aie.objectfifo.core_endpoint @of0_cons(%tile13) drains @of0_pool
  }
}

//===----------------------------------------------------------------------===//
// STAGE 3 -- after --aie-objectfifo-lower-dmas     (no change)
//
// There is no DMA endpoint, so this pass has nothing to match.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STAGE 4 -- after --aie-objectfifo-lower-cores
//
// Both cores rotate through the same four buffers and share the same lock pair.
// The core at @of0_prod acquires the produceLock and releases the consumeLock;
// the core at @of0_cons does the reverse. Both endpoints and the pool are then
// erased.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SAME-CORE FIFO
//
// test/objectFifo-stateful-transform/access_patterns/same_core_producer_consumer_test.mlir
// declares aie.objectfifo @of (%tile12, {%tile12}, 3) and has one core alternate
// between filling and draining it. Both endpoints land on the same tile, over
// one pool:
//
//   aie.objectfifo.pool @of_pool(%tile12) {depth = 3 : i32, ...} : memref<16xi32>
//   aie.objectfifo.core_endpoint @of_fill(%tile12)  fills  @of_pool
//   aie.objectfifo.core_endpoint @of_drain(%tile12) drains @of_pool
//
//   %core12 = aie.core(%tile12) {
//     %sv0 = aie.objectfifo.acquire @of_fill (2) : ...
//     ...
//     aie.objectfifo.release @of_fill [1]
//     %sv1 = aie.objectfifo.acquire @of_drain (1) : ...
//     ...
//     aie.objectfifo.release @of_drain [1]
//   }
//
// The core names whichever endpoint it is acting through, so its acquire and
// release ops carry no direction of their own.
//===----------------------------------------------------------------------===//
