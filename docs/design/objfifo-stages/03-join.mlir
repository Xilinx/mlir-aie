//===- 03-join.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESIGN SKETCH -- not a lit test. See 00-model.mlir first.
//
// Three fifos joined into one on a MemTile, then out to the shim. Derived from
// test/objectFifo-stateful-transform/data_movement_patterns/link/
// link_test_join_offsets.mlir.
//
// A join is one pool with several segments, each filled by its own endpoint and
// all drained by one.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STAGE 0 -- input
//===----------------------------------------------------------------------===//

module @stage0 {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)   // shim
    %tile21 = aie.tile(2, 1)   // memtile
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
  }
}

//===----------------------------------------------------------------------===//
// STAGE 1 -- after --aie-objectfifo-split
//
// The MemTile carries ONE pool of three segments. Three DMA endpoints fill it,
// one per segment; a fourth drains all three. The source tiles have ordinary
// single-segment pools of their own, and the shim endpoint has no pool.
//===----------------------------------------------------------------------===//

module @stage1 {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo.pool @mt_pool(%tile21) {
      depth    = 2 : i32,
      segments = [<offset = 0,  size = 16>,
                  <offset = 16, size = 20>,
                  <offset = 36, size = 12>]
    } : memref<48xi32>

    aie.objectfifo.pool @link1_pool(%tile22) {depth = 2 : i32, segments = [<offset = 0, size = 16>]} : memref<4x4xi32>
    aie.objectfifo.pool @link2_pool(%tile23) {depth = 2 : i32, segments = [<offset = 0, size = 20>]} : memref<20xi32>
    aie.objectfifo.pool @link3_pool(%tile33) {depth = 2 : i32, segments = [<offset = 0, size = 12>]} : memref<12xi32>

    aie.objectfifo.core_endpoint @link1_core(%tile22) fills  @link1_pool
    aie.objectfifo.dma_endpoint  @link1_out(%tile22)  drains @link1_pool
    aie.objectfifo.core_endpoint @link2_core(%tile23) fills  @link2_pool
    aie.objectfifo.dma_endpoint  @link2_out(%tile23)  drains @link2_pool
    aie.objectfifo.core_endpoint @link3_core(%tile33) fills  @link3_pool
    aie.objectfifo.dma_endpoint  @link3_out(%tile33)  drains @link3_pool

    aie.objectfifo.dma_endpoint @mt_in1(%tile21) fills  @mt_pool {segments = [0]}
    aie.objectfifo.dma_endpoint @mt_in2(%tile21) fills  @mt_pool {segments = [1]}
    aie.objectfifo.dma_endpoint @mt_in3(%tile21) fills  @mt_pool {segments = [2]}
    aie.objectfifo.dma_endpoint @mt_out(%tile21) drains @mt_pool

    aie.objectfifo.dma_endpoint @shim_in(%tile20) {}

    aie.objectfifo.flow from @link1_out to [@mt_in1]
    aie.objectfifo.flow from @link2_out to [@mt_in2]
    aie.objectfifo.flow from @link3_out to [@mt_in3]
    aie.objectfifo.flow from @mt_out   to [@shim_in]
  }
}

//===----------------------------------------------------------------------===//
// STAGE 2 -- after --aie-objectfifo-allocate
//
// The MemTile pool gets two buffers sized for the whole object and one lock pair
// per segment. Channels and flows follow, and the shim endpoint becomes a
// shim allocation.
//===----------------------------------------------------------------------===//

module @stage2 {
  aie.device(xcve2302) {
    %j0 = aie.buffer(%tile21) {sym_name = "mt_buff_0"} : memref<48xi32>
    %j1 = aie.buffer(%tile21) {sym_name = "mt_buff_1"} : memref<48xi32>
    %p0 = aie.lock(%tile21) {init = 2 : i32, sym_name = "mt_prod_lock_0"}
    %c0 = aie.lock(%tile21) {init = 0 : i32, sym_name = "mt_cons_lock_0"}
    %p1 = aie.lock(%tile21) {init = 2 : i32, sym_name = "mt_prod_lock_1"}
    %c1 = aie.lock(%tile21) {init = 0 : i32, sym_name = "mt_cons_lock_1"}
    %p2 = aie.lock(%tile21) {init = 2 : i32, sym_name = "mt_prod_lock_2"}
    %c2 = aie.lock(%tile21) {init = 0 : i32, sym_name = "mt_cons_lock_2"}

    aie.objectfifo.pool @mt_pool(%tile21) {
      depth    = 2 : i32,
      buffers  = [@mt_buff_0, @mt_buff_1],
      segments = [<produceLock = @mt_prod_lock_0, consumeLock = @mt_cons_lock_0,
                   offset = 0,  size = 16>,
                  <produceLock = @mt_prod_lock_1, consumeLock = @mt_cons_lock_1,
                   offset = 16, size = 20>,
                  <produceLock = @mt_prod_lock_2, consumeLock = @mt_cons_lock_2,
                   offset = 36, size = 12>]
    } : memref<48xi32>

    // source-tile pools are ordinary and private, one per tile

    aie.objectfifo.dma_endpoint @mt_in1(%tile21) fills  @mt_pool {segments = [0], channel = S2MM 0}
    aie.objectfifo.dma_endpoint @mt_in2(%tile21) fills  @mt_pool {segments = [1], channel = S2MM 1}
    aie.objectfifo.dma_endpoint @mt_in3(%tile21) fills  @mt_pool {segments = [2], channel = S2MM 2}
    aie.objectfifo.dma_endpoint @mt_out(%tile21) drains @mt_pool {channel = MM2S 0}

    aie.flow(%tile22, DMA : 0, %tile21, DMA : 0)
    aie.flow(%tile23, DMA : 0, %tile21, DMA : 1)
    aie.flow(%tile33, DMA : 0, %tile21, DMA : 2)
    aie.flow(%tile21, DMA : 0, %tile20, DMA : 0)

    aie.shim_dma_allocation @link4_shim_alloc(%tile20, S2MM, 0)
  }
}

//===----------------------------------------------------------------------===//
// STAGE 3 -- after --aie-objectfifo-lower-dmas
//
// BD emission walks buffers, then the segments each endpoint selects:
//
//   @mt_in1: 2 buffers x 1 segment  = 2 BDs on S2MM 0
//   @mt_in2: 2 buffers x 1 segment  = 2 BDs on S2MM 1
//   @mt_in3: 2 buffers x 1 segment  = 2 BDs on S2MM 2
//   @mt_out: 2 buffers x 3 segments = 6 BDs on MM2S 0
//
// Filling endpoints acquire the produceLock and release the consumeLock;
// @mt_out, draining, does the reverse for each segment it covers.
//===----------------------------------------------------------------------===//

module @stage3 {
  aie.device(xcve2302) {
    %memtile_dma_2_1 = aie.memtile_dma(%tile21) {
      aie.dma_start(S2MM, 0, ^bb1, ^bb3)          // @mt_in1
    ^bb1:
      aie.use_lock(%p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%j0 : memref<48xi32>, 0, 16)
      aie.use_lock(%c0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:
      aie.use_lock(%p0, AcquireGreaterEqual, 1)
      aie.dma_bd(%j1 : memref<48xi32>, 0, 16)
      aie.use_lock(%c0, Release, 1)
      aie.next_bd ^bb1

    ^bb3:
      aie.dma_start(S2MM, 1, ^bb4, ^bb6)          // @mt_in2
    ^bb4:
      aie.use_lock(%p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%j0 : memref<48xi32>, 16, 20)
      aie.use_lock(%c1, Release, 1)
      aie.next_bd ^bb5
    ^bb5:
      aie.use_lock(%p1, AcquireGreaterEqual, 1)
      aie.dma_bd(%j1 : memref<48xi32>, 16, 20)
      aie.use_lock(%c1, Release, 1)
      aie.next_bd ^bb4

    ^bb6:
      aie.dma_start(S2MM, 2, ^bb7, ^bb9)          // @mt_in3
    ^bb7:
      aie.use_lock(%p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%j0 : memref<48xi32>, 36, 12)
      aie.use_lock(%c2, Release, 1)
      aie.next_bd ^bb8
    ^bb8:
      aie.use_lock(%p2, AcquireGreaterEqual, 1)
      aie.dma_bd(%j1 : memref<48xi32>, 36, 12)
      aie.use_lock(%c2, Release, 1)
      aie.next_bd ^bb7

    ^bb9:
      aie.dma_start(MM2S, 0, ^bb10, ^bb16)        // @mt_out
    ^bb10:
      aie.use_lock(%c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%j0 : memref<48xi32>, 0, 16)
      aie.use_lock(%p0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:
      aie.use_lock(%c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%j0 : memref<48xi32>, 16, 20)
      aie.use_lock(%p1, Release, 1)
      aie.next_bd ^bb12
    ^bb12:
      aie.use_lock(%c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%j0 : memref<48xi32>, 36, 12)
      aie.use_lock(%p2, Release, 1)
      aie.next_bd ^bb13
    ^bb13:
      aie.use_lock(%c0, AcquireGreaterEqual, 1)
      aie.dma_bd(%j1 : memref<48xi32>, 0, 16)
      aie.use_lock(%p0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:
      aie.use_lock(%c1, AcquireGreaterEqual, 1)
      aie.dma_bd(%j1 : memref<48xi32>, 16, 20)
      aie.use_lock(%p1, Release, 1)
      aie.next_bd ^bb15
    ^bb15:
      aie.use_lock(%c2, AcquireGreaterEqual, 1)
      aie.dma_bd(%j1 : memref<48xi32>, 36, 12)
      aie.use_lock(%p2, Release, 1)
      aie.next_bd ^bb10
    ^bb16:
      aie.end
    }
  }
}

//===----------------------------------------------------------------------===//
// JOIN WITH CORE PARTICIPANTS
//
// Nothing in a segment names the kind of actor that handles it, so a join whose
// participants are cores on memory-adjacent tiles is the same pool with core
// endpoints in place of DMA endpoints, and no channels or BD chains at all:
//
//   aie.objectfifo.core_endpoint @a(%t22) fills  @mt_pool {segments = [0]}
//   aie.objectfifo.core_endpoint @b(%t23) fills  @mt_pool {segments = [1]}
//   aie.objectfifo.core_endpoint @c(%t21) drains @mt_pool
//
// @a and @b each hold a strict slice of the shared buffers, so their objects are
// memref.subviews at the segment offset. @c covers every segment, whose union is
// the whole buffer, so it receives the buffer itself once it has acquired all
// the segment locks.
//===----------------------------------------------------------------------===//
