//===- dma_out_of_order.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Arm an out-of-order S2MM channel on the libxaie path too.

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// The receive BDs do not chain (use_next_bd=0 == the last SetNextBd arg is 0).
// CHECK: XAie_DmaSetNextBd({{.*}}, {{.*}}, {{.*}}0));
// Each receive BD must be packet-enabled.
// CHECK: XAie_DmaSetPkt(
// Out-of-order mode must be armed while idle (e.g., before it is enabled).
// CHECK: XAie_DmaChannelDescInit(ctx->XAieDevInst, &ooo_desc_0_2_0, XAie_TileLoc(0,2))
// CHECK: XAie_DmaChannelEnOutofOrder(&ooo_desc_0_2_0, XAIE_ENABLE)
// CHECK: XAie_DmaWriteChannel(ctx->XAieDevInst, &ooo_desc_0_2_0, XAie_TileLoc(0,2),{{.*}}0,{{.*}}DMA_S2MM)
// CHECK: XAie_DmaDeclareQueueConfig(ooo_queue_0_2_0, {{.*}}0, {{.*}}3, {{.*}}XAIE_DISABLE, {{.*}}XAIE_ENABLE)
// CHECK: XAie_DmaChannelSetStartQueueGeneric(ctx->XAieDevInst, XAie_TileLoc(0,2),{{.*}}0,{{.*}}DMA_S2MM, &ooo_queue_0_2_0)

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) : memref<8xi32>
    aie.mem(%t) {
        aie.dma_start(S2MM, 0, ^bd0, ^end, repeat_count = 2) { out_of_order }
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
  }
}
