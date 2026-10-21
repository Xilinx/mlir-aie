//===- shim_dma_packet_attr.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// The packet header may be given as the dma_bd's `packet` attribute instead of
// a separate aie.dma_bd_packet op.

// CHECK-LABEL: int mlir_aie_configure_shimdma_70(aie_libxaie_ctx_t* ctx) {
// CHECK: XAie_DmaSetPkt(&(dma_tile70_bd0), XAie_PacketInit(2,5))
// CHECK: XAie_DmaWriteBd(ctx->XAieDevInst, &(dma_tile70_bd0), XAie_TileLoc(7,0),  /* bd */ 0)

module {
 aie.device(xcvc1902) {
  %buf = aie.external_buffer { sym_name = "buf" } : memref<32x32xi32>

  %tile70 = aie.tile(7, 0)
  %lock70 = aie.lock(%tile70, 0)

  %shimdma70 = aie.shim_dma(%tile70)  {
    aie.dma_start(MM2S, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%lock70, Acquire, %c1_ul1)
    aie.dma_bd(%buf : memref<32x32xi32> offset = 0 len = 1024) {packet = #aie.packet_info<pkt_type = 5, pkt_id = 2>}
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%lock70, Release, %c0_ul2)
    aie.next_bd ^bb1
  ^bb2:  // pred: ^bb0
    aie.end
  }
 }
}
