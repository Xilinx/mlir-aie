//===- packet_drop_header_shim.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-packet-flows --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// CHECK: void mlir_aie_configure_switchboxes(aie_libxaie_ctx_t* ctx) {
// CHECK: // Core Stream Switch column 7 row 0
// CHECK: XAieTile_StrmConfigMstr(&(ctx->TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK:     XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK:     XAIE_ENABLE /*drop_header*/,

//
// This tests the switchbox configuration lowering for packet switched routing
// to drop headers when the packet's destination is a DMA.
//
module @aie_module  {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 2)

  %10 = AIE.lock(%t71, 1)
  %lock1 = AIE.lock(%t70, 1)
  %lock2 = AIE.lock(%t70, 2)

  %11 = AIE.buffer(%t71) {sym_name = "buf1"} : memref<16xi32, 2>
  %buf_i = AIE.external_buffer 0x020100004000 : memref<16xi32>
  %buf_o = AIE.external_buffer 0x020100004020 : memref<16xi32>

  %12 = AIE.mem(%t71)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    %dstDma = AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2:
    AIE.useLock(%10, Acquire, 0)
    AIE.dmaBdPacket(0x2, 3)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 1)
    cf.br ^bb2
  ^bb3:
    AIE.useLock(%10, Acquire, 1)
    AIE.dmaBdPacket(0x6, 10)
    AIE.dmaBd(<%11 : memref<16xi32, 2>, 0, 16>, 0)
    AIE.useLock(%10, Release, 0)
    cf.br ^bb3
  ^end:
    AIE.end
  }

  %dma = AIE.shimDMA(%t70)  {
    AIE.dmaStart("S2MM0", ^bb2, ^dma0)
  ^dma0:
    AIE.dmaStart("MM2S0", ^bb3, ^end)
  ^bb2:
    AIE.useLock(%lock1, Acquire, 1)
    AIE.dmaBdPacket(0x2, 3)
    AIE.dmaBd(<%buf_i : memref<16xi32>, 0, 16>, 0)
    AIE.useLock(%lock1, Release, 0)
    cf.br ^bb2
  ^bb3:
    AIE.useLock(%lock2, Acquire, 1)
    AIE.dmaBdPacket(0x6, 10)
    AIE.dmaBd(<%buf_o : memref<16xi32>, 0, 16>, 0)
    AIE.useLock(%lock2, Release, 0)
    cf.br ^bb3
  ^end:
    AIE.end
  }
  AIE.packet_flow(0x3) {
    AIE.packet_source<%t70, DMA : 0>
    AIE.packet_dest<%t71, DMA : 0>
  }
  AIE.packet_flow(0xA) {
    AIE.packet_source<%t71, DMA : 0>
    AIE.packet_dest<%t70, DMA : 0>
  }
}
