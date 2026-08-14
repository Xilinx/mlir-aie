//===- 01-simple-dma.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESIGN SKETCH -- not a lit test. See 00-model.mlir first.
//
// One producer tile, one consumer tile, not memory-adjacent, depth 2 ping-pong.
// Derived from test/objectFifo-stateful-transform/base/base_test_AIE2.mlir (@of1).
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STAGE 0 -- input
//===----------------------------------------------------------------------===//

module @stage0 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %v  = arith.constant 42 : i32
      %sv = aie.objectfifo.acquire @of1 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %e  = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      memref.store %v, %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of1 (Produce, 1)
      aie.end
    }

    %core33 = aie.core(%tile33) {
      %c0 = arith.constant 0 : index
      %sv = aie.objectfifo.acquire @of1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %e  = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %v  = memref.load %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of1 (Consume, 1)
      aie.end
    }
  }
}

//===----------------------------------------------------------------------===//
// STAGE 1 -- after --aie-objectfifo-split
//
// Two pools, one per tile, each with four actors' worth of structure: a core and
// a DMA at each end. Buffers and locks are not assigned yet; the segment
// boundaries are, since they follow from the element type.
//
// The two ends of a tile's pool are opposite: on tile12 the core fills and the
// DMA drains, on tile33 the DMA fills and the core drains.
//===----------------------------------------------------------------------===//

module @stage1 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo.pool @of1_prod_pool(%tile12) {
      depth = 2 : i32, segments = [<offset = 0, size = 16>]
    } : memref<16xi32>

    aie.objectfifo.pool @of1_cons_pool(%tile33) {
      depth = 2 : i32, segments = [<offset = 0, size = 16>]
    } : memref<16xi32>

    aie.objectfifo.core_endpoint @of1_prod_core(%tile12) fills  @of1_prod_pool
    aie.objectfifo.dma_endpoint  @of1_prod_dma(%tile12)  drains @of1_prod_pool
    aie.objectfifo.dma_endpoint  @of1_cons_dma(%tile33)  fills  @of1_cons_pool
    aie.objectfifo.core_endpoint @of1_cons_core(%tile33) drains @of1_cons_pool

    aie.objectfifo.flow from @of1_prod_dma to [@of1_cons_dma]

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %v  = arith.constant 42 : i32
      %sv = aie.objectfifo.acquire @of1_prod_core (1) : !aie.objectfifosubview<memref<16xi32>>
      %e  = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      memref.store %v, %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of1_prod_core (1)
      aie.end
    }

    %core33 = aie.core(%tile33) {
      %c0 = arith.constant 0 : index
      %sv = aie.objectfifo.acquire @of1_cons_core (1) : !aie.objectfifosubview<memref<16xi32>>
      %e  = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %v  = memref.load %e[%c0] : memref<16xi32>
      aie.objectfifo.release @of1_cons_core (1)
      aie.end
    }
  }
}

//===----------------------------------------------------------------------===//
// STAGE 2 -- after --aie-objectfifo-allocate
//
// Buffers, locks, channels and the flow. Each pool gets its own resources, the
// tiles not being memory-adjacent.
//
// A pool that already names buffers and locks, or an endpoint that already names
// a channel, is left as written.
//===----------------------------------------------------------------------===//

module @stage2 {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    %b0 = aie.buffer(%tile12) {sym_name = "of1_buff_0"} : memref<16xi32>
    %b1 = aie.buffer(%tile12) {sym_name = "of1_buff_1"} : memref<16xi32>
    %pl = aie.lock(%tile12) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
    %cl = aie.lock(%tile12) {init = 0 : i32, sym_name = "of1_cons_lock_0"}

    %cb0 = aie.buffer(%tile33) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
    %cb1 = aie.buffer(%tile33) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
    %cpl = aie.lock(%tile33) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
    %ccl = aie.lock(%tile33) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}

    aie.objectfifo.pool @of1_prod_pool(%tile12) {
      depth    = 2 : i32,
      buffers  = [@of1_buff_0, @of1_buff_1],
      segments = [<produceLock = @of1_prod_lock_0, consumeLock = @of1_cons_lock_0,
                   offset = 0, size = 16>]
    } : memref<16xi32>

    aie.objectfifo.pool @of1_cons_pool(%tile33) {
      depth    = 2 : i32,
      buffers  = [@of1_cons_buff_0, @of1_cons_buff_1],
      segments = [<produceLock = @of1_cons_prod_lock_0, consumeLock = @of1_cons_cons_lock_0,
                   offset = 0, size = 16>]
    } : memref<16xi32>

    aie.objectfifo.core_endpoint @of1_prod_core(%tile12) fills  @of1_prod_pool
    aie.objectfifo.dma_endpoint  @of1_prod_dma(%tile12)  drains @of1_prod_pool {channel = MM2S 0}
    aie.objectfifo.dma_endpoint  @of1_cons_dma(%tile33)  fills  @of1_cons_pool  {channel = S2MM 0}
    aie.objectfifo.core_endpoint @of1_cons_core(%tile33) drains @of1_cons_pool

    aie.flow(%tile12, DMA : 0, %tile33, DMA : 0)
  }
}

//===----------------------------------------------------------------------===//
// STAGE 3 -- after --aie-objectfifo-lower-dmas
//
// @of1_prod_dma drains, so it acquires the consumeLock and releases the
// produceLock; @of1_cons_dma fills, so it does the reverse. Both DMA endpoints
// are erased; the pools remain because their core endpoints do.
//===----------------------------------------------------------------------===//

module @stage3 {
  aie.device(xcve2302) {
    aie.flow(%tile12, DMA : 0, %tile33, DMA : 0)

    %mem12 = aie.mem(%tile12) {
      aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%cl, AcquireGreaterEqual, 1)
      aie.dma_bd(%b0 : memref<16xi32>, 0, 16)
      aie.use_lock(%pl, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%cl, AcquireGreaterEqual, 1)
      aie.dma_bd(%b1 : memref<16xi32>, 0, 16)
      aie.use_lock(%pl, Release, 1)
      aie.next_bd ^bb1
    ^bb3:
      aie.end
    }

    %mem33 = aie.mem(%tile33) {
      aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:
      aie.use_lock(%cpl, AcquireGreaterEqual, 1)
      aie.dma_bd(%cb0 : memref<16xi32>, 0, 16)
      aie.use_lock(%ccl, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%cpl, AcquireGreaterEqual, 1)
      aie.dma_bd(%cb1 : memref<16xi32>, 0, 16)
      aie.use_lock(%ccl, Release, 1)
      aie.next_bd ^bb1
    ^bb3:
      aie.end
    }
  }
}

//===----------------------------------------------------------------------===//
// STAGE 4 -- after --aie-objectfifo-lower-cores
//
// The core at @of1_cons_core drains, so it acquires the consumeLock and releases
// the produceLock. The core endpoints and both pools are erased, leaving plain
// AIE IR.
//
// This pass also annotates aie.unroll_hint and promotes the bookkeeping allocas
// to SSA.
//===----------------------------------------------------------------------===//

module @stage4 {
  aie.device(xcve2302) {
    %core33 = aie.core(%tile33) {
      %c0     = arith.constant 0 : index
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %delta = arith.subi %c1_i32, %c0_i32 : i32
      %acq   = arith.maxsi %delta, %c0_i32 : i32
      aie.use_lock(%ccl, AcquireGreaterEqual, %acq)
      %v = memref.load %cb0[%c0] : memref<16xi32>
      aie.use_lock(%cpl, Release, %c1_i32)
      aie.end
    }
  }
}
