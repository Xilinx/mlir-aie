//===- 04-distribute.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESIGN SKETCH -- not a lit test. See 00-model.mlir first.
//
// One fifo split into three on a MemTile. A distribute is one pool with several
// segments, all filled by one endpoint and each drained by its own. Compare
// test/objectFifo-stateful-transform/data_movement_patterns/link/
// link_test_distribute_offsets.mlir.
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

    aie.objectfifo @in   (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @out1 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out2 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @out3 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@in] -> [@out1, @out2, @out3] ([][0, 16, 36])
  }
}

//===----------------------------------------------------------------------===//
// STAGE 1 -- after --aie-objectfifo-split
//===----------------------------------------------------------------------===//

module @stage1 {
  aie.device(xcve2302) {
    aie.objectfifo.pool @mt_pool(%tile21) {
      depth    = 2 : i32,
      segments = [<offset = 0,  size = 16>,
                  <offset = 16, size = 20>,
                  <offset = 36, size = 12>]
    } : memref<48xi32>

    aie.objectfifo.pool @out1_pool(%tile22) {depth = 2 : i32, segments = [<offset = 0, size = 16>]} : memref<16xi32>
    aie.objectfifo.pool @out2_pool(%tile23) {depth = 2 : i32, segments = [<offset = 0, size = 20>]} : memref<20xi32>
    aie.objectfifo.pool @out3_pool(%tile33) {depth = 2 : i32, segments = [<offset = 0, size = 12>]} : memref<12xi32>

    aie.objectfifo.dma_endpoint @shim_out(%tile20) {}

    aie.objectfifo.dma_endpoint @mt_in(%tile21)   fills  @mt_pool
    aie.objectfifo.dma_endpoint @mt_out1(%tile21) drains @mt_pool {segments = [0]}
    aie.objectfifo.dma_endpoint @mt_out2(%tile21) drains @mt_pool {segments = [1]}
    aie.objectfifo.dma_endpoint @mt_out3(%tile21) drains @mt_pool {segments = [2]}

    aie.objectfifo.dma_endpoint  @out1_in(%tile22)   fills  @out1_pool
    aie.objectfifo.core_endpoint @out1_core(%tile22) drains @out1_pool
    aie.objectfifo.dma_endpoint  @out2_in(%tile23)   fills  @out2_pool
    aie.objectfifo.core_endpoint @out2_core(%tile23) drains @out2_pool
    aie.objectfifo.dma_endpoint  @out3_in(%tile33)   fills  @out3_pool
    aie.objectfifo.core_endpoint @out3_core(%tile33) drains @out3_pool

    aie.objectfifo.flow from @shim_out to [@mt_in]
    aie.objectfifo.flow from @mt_out1  to [@out1_in]
    aie.objectfifo.flow from @mt_out2  to [@out2_in]
    aie.objectfifo.flow from @mt_out3  to [@out3_in]
  }
}

//===----------------------------------------------------------------------===//
// STAGE 2 -- after --aie-objectfifo-allocate
//
// The MemTile pool is sized for the whole object and carries one lock pair per
// segment. The endpoint that fills covers every segment; each draining endpoint
// covers one.
//===----------------------------------------------------------------------===//

module @stage2 {
  aie.device(xcve2302) {
    %d0 = aie.buffer(%tile21) {sym_name = "mt_buff_0"} : memref<48xi32>
    %d1 = aie.buffer(%tile21) {sym_name = "mt_buff_1"} : memref<48xi32>
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

    aie.objectfifo.dma_endpoint @mt_in(%tile21)   fills  @mt_pool {channel = S2MM 0}
    aie.objectfifo.dma_endpoint @mt_out1(%tile21) drains @mt_pool {segments = [0], channel = MM2S 0}
    aie.objectfifo.dma_endpoint @mt_out2(%tile21) drains @mt_pool {segments = [1], channel = MM2S 1}
    aie.objectfifo.dma_endpoint @mt_out3(%tile21) drains @mt_pool {segments = [2], channel = MM2S 2}

    aie.shim_dma_allocation @in_shim_alloc(%tile20, MM2S, 0)
  }
}

//===----------------------------------------------------------------------===//
// STAGE 3 -- after --aie-objectfifo-lower-dmas
//
//   @mt_in:   2 buffers x 3 segments = 6 BDs on S2MM 0  (fills)
//   @mt_out1: 2 buffers x 1 segment  = 2 BDs on MM2S 0  (offset 0,  size 16)
//   @mt_out2: 2 buffers x 1 segment  = 2 BDs on MM2S 1  (offset 16, size 20)
//   @mt_out3: 2 buffers x 1 segment  = 2 BDs on MM2S 2  (offset 36, size 12)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DISTRIBUTE WITH CORE PARTICIPANTS
//
// Replacing the draining DMA endpoints with core endpoints on memory-adjacent
// tiles gives a distribute with no channels and no BD chains. Each such core
// holds a strict slice of the shared buffers, so its object is a memref.subview
// at the segment offset.
//===----------------------------------------------------------------------===//
