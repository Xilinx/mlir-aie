//===- dma_out_of_order.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// The receive BDs do not chain (use_next_bd=0 == SetNextBd=0), and a
// non-iterating BD gets no iteration; SetAddrLen is followed directly by
// SetNextBd (and not SetBdIteration).
// CHECK: XAie_DmaSetAddrLen(&(dma_tile02_bd0)
// CHECK-NEXT: XAie_DmaSetNextBd(&(dma_tile02_bd0){{.*}}0));

// Each receive BD must be packet-enabled.
// CHECK: XAie_DmaSetPkt(

// Out-of-order mode must be armed while idle (e.g., before enabling).
// CHECK: XAie_DmaChannelDescInit(ctx->XAieDevInst, &ooo_desc_0_2_0, XAie_TileLoc(0,2))
// CHECK: XAie_DmaChannelEnOutofOrder(&ooo_desc_0_2_0, XAIE_ENABLE)
// CHECK: XAie_DmaWriteChannel(ctx->XAieDevInst, &ooo_desc_0_2_0, XAie_TileLoc(0,2),{{.*}}0,{{.*}}DMA_S2MM)
// CHECK: XAie_DmaDeclareQueueConfig(ooo_queue_0_2_0, {{.*}}0, {{.*}}6, {{.*}}XAIE_DISABLE, {{.*}}XAIE_ENABLE)
// CHECK: XAie_DmaChannelSetStartQueueGeneric(ctx->XAieDevInst, XAie_TileLoc(0,2),{{.*}}0,{{.*}}DMA_S2MM, &ooo_queue_0_2_0)

// Tile (1,2)'s receive BD also iterates (an m-packet BD).
// CHECK: XAie_DmaSetBdIteration(&(dma_tile12_bd0), 4, 2, 0)
// CHECK: XAie_DmaChannelEnOutofOrder(&ooo_desc_1_2_0, XAIE_ENABLE)

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 5) { out_of_order }
      ^bd0:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32 }
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b : memref<8xi32> offset = 4 len = 4) { bd_id = 1 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
    %t12 = aie.tile(1, 2)
    %b12 = aie.buffer(%t12) : memref<8xi32>
    aie.mem(%t12) {
        aie.dma_start(S2MM, 0, ^i0, ^end2, repeat_count = 7) { out_of_order }
      ^i0:
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%b12 : memref<8xi32> offset = 0 len = 4) { bd_id = 0 : i32, iteration = #aie.bd_iteration<size = 2, stride = 4, current = 0> }
        aie.next_bd ^end2
      ^end2:
        aie.end
    }
  }
}
