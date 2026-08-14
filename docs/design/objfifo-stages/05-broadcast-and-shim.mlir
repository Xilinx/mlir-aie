//===- 05-broadcast-and-shim.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DESIGN SKETCH -- not a lit test. See 00-model.mlir first.
//
// Broadcast: one source channel feeding several destinations. Shim endpoints:
// the boundary to DDR, where there are no buffers and no locks.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// STAGE 0 -- input
//===----------------------------------------------------------------------===//

module @stage0 {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t02  = aie.tile(0, 2)
    %t03  = aie.tile(0, 3)

    aie.objectfifo @in (%shim, {%t02}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @bc (%t02, {%t03, %shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}

//===----------------------------------------------------------------------===//
// STAGE 1 -- after --aie-objectfifo-split
//
// Each consumer tile of a broadcast gets its own pool, with its own buffers and
// locks; one flow lists every destination, matching the one source channel that
// feeds the multicast route.
//
// The two shim endpoints have no pool.
//===----------------------------------------------------------------------===//

module @stage1 {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t02  = aie.tile(0, 2)
    %t03  = aie.tile(0, 3)

    aie.objectfifo.pool @in_pool(%t02)     {depth = 2 : i32, segments = [<offset = 0, size = 16>]} : memref<16xi32>
    aie.objectfifo.pool @bc_pool(%t02)     {depth = 2 : i32, segments = [<offset = 0, size = 16>]} : memref<16xi32>
    aie.objectfifo.pool @bc_c0_pool(%t03)  {depth = 2 : i32, segments = [<offset = 0, size = 16>]} : memref<16xi32>

    aie.objectfifo.dma_endpoint @in_shim(%shim) {}

    aie.objectfifo.dma_endpoint  @in_dma(%t02)   fills  @in_pool
    aie.objectfifo.core_endpoint @in_core(%t02)  drains @in_pool

    aie.objectfifo.core_endpoint @bc_core(%t02)  fills  @bc_pool
    aie.objectfifo.dma_endpoint  @bc_out(%t02)   drains @bc_pool

    aie.objectfifo.dma_endpoint  @bc_c0_in(%t03)   fills  @bc_c0_pool
    aie.objectfifo.core_endpoint @bc_c0_core(%t03) drains @bc_c0_pool

    aie.objectfifo.dma_endpoint @bc_c1_shim(%shim) {}

    aie.objectfifo.flow from @in_shim to [@in_dma]
    aie.objectfifo.flow from @bc_out  to [@bc_c0_in, @bc_c1_shim]
  }
}

//===----------------------------------------------------------------------===//
// STAGE 2 -- after --aie-objectfifo-allocate
//
// Shim endpoints receive a channel and become shim allocations, which is what
// the runtime sequence resolves to find the tile, direction and channel to
// drive. Every other endpoint gets its resources as usual.
//===----------------------------------------------------------------------===//

module @stage2 {
  aie.device(npu1) {
    %cb0 = aie.buffer(%t02) {sym_name = "in_buff_0"} : memref<16xi32>
    %cb1 = aie.buffer(%t02) {sym_name = "in_buff_1"} : memref<16xi32>
    %cpl = aie.lock(%t02) {init = 2 : i32, sym_name = "in_prod_lock_0"}
    %ccl = aie.lock(%t02) {init = 0 : i32, sym_name = "in_cons_lock_0"}

    aie.objectfifo.pool @in_pool(%t02) {
      depth    = 2 : i32,
      buffers  = [@in_buff_0, @in_buff_1],
      segments = [<produceLock = @in_prod_lock_0, consumeLock = @in_cons_lock_0,
                   offset = 0, size = 16>]
    } : memref<16xi32>

    aie.objectfifo.dma_endpoint  @in_dma(%t02)  fills  @in_pool {channel = S2MM 0}
    aie.objectfifo.core_endpoint @in_core(%t02) drains @in_pool

    aie.flow(%shim, DMA : 0, %t02,  DMA : 0)
    aie.flow(%t02,  DMA : 0, %t03,  DMA : 0)
    aie.flow(%t02,  DMA : 0, %shim, DMA : 0)

    aie.shim_dma_allocation @in_shim_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @bc_shim_alloc(%shim, S2MM, 0)
  }
}

//===----------------------------------------------------------------------===//
// STAGE 3 -- after --aie-objectfifo-lower-dmas
//
// Endpoints with a pool emit their BD chains. The shim endpoints have none to
// emit; their record is the shim allocation, and the runtime sequence programs
// the transfer itself.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EXTERNAL BUFFERS
//
// A shim end that registers external buffers has both buffers and locks, and so
// is an ordinary pool whose buffers happen to be aie.external_buffer ops. It
// emits an aie.shim_dma BD chain like any other DMA endpoint, alongside its
// shim allocation.
//===----------------------------------------------------------------------===//
