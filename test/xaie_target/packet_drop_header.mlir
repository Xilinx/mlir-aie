//===- packet_drop_header.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-packet-flows --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// CHECK: void mlir_aie_configure_switchboxes(aie_libxaie_ctx_t* ctx) {
// CHECK: // Core Stream Switch column 7 row 1
// CHECK: XAieTile_StrmConfigMstr(&(ctx->TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK:     XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK:     XAIE_ENABLE /*drop_header*/,
// CHECK: XAieTile_StrmConfigMstr(&(ctx->TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_MPORT_SOUTH(&(ctx->TileInst[x][y]), 0),
// CHECK:     XAIETILE_STRSW_MPORT_SOUTH(&(ctx->TileInst[x][y]), 0),
// CHECK:     XAIE_DISABLE /*drop_header*/,

//
// This tests the switchbox configuration lowering for packet switched routing
// to drop headers when the packet's destination is a DMA.
//
module @aie_module  {
 AIE.device(xcvc1902) {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)

  %10 = AIE.lock(%t71, 1)
  %11 = AIE.buffer(%t71) {address = 3072 : i32, sym_name = "buf1"} : memref<16xi32, 2>

  %12 = AIE.mem(%t71)  {
    %srcDma = AIE.dmaStart("S2MM", 0, ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S", 0, ^bb3, ^end)
  ^bb2:
    AIE.useLock(%10, Acquire, 0)
    AIE.dmaBdPacket(0x2, 3)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 1)
    AIE.nextBd ^bb2
  ^bb3:
    AIE.useLock(%10, Acquire, 1)
    AIE.dmaBdPacket(0x6, 10)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 0)
    AIE.nextBd ^bb3
  ^end:
    AIE.end
  }
  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, South : 4>
    AIE.packet_dest<%t71, DMA : 0>
  }
  AIE.packet_flow(0xA) {
    AIE.packet_source<%t71, DMA : 0>
    AIE.packet_dest<%t70, South : 0>
  }
 }
}
