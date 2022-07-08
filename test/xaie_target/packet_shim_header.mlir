//===- packet_shim_header.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-packet-flows --aie-assign-buffer-addresses %s | aie-translate --aie-generate-xaie | FileCheck %s

// CHECK: void mlir_aie_configure_dmas(aie_libxaie_ctx_t* ctx) {
// CHECK: XAieDma_ShimBdSetNext(&ShimDMAInst_7_0,  /* bd */ 0,  /* nextbd */ 0);
// CHECK: XAieDma_ShimBdSetPkt(&ShimDMAInst_7_0,  /* bd */ 0,  /* en */ 1,  /* type */ 6,  /* id */ 10);

//
// This tests the shim DMA BD configuration lowering for packet switched routing
// to insert packet headers for shim DMA BDs.
//
module @aie_module  {
  %t70 = AIE.tile(7, 0)
  %t71 = AIE.tile(7, 1)

  %10 = AIE.lock(%t71, 1)
  %11 = AIE.buffer(%t71) {sym_name = "buf1"} : memref<32xi32, 2>
  %buffer = AIE.external_buffer 0x020100004000 : memref<32xi32>

  %12 = AIE.mem(%t71)  {
    %srcDma = AIE.dmaStart("S2MM0", ^bb2, ^end)
  ^bb2:
    AIE.useLock(%10, Acquire, 0)
    AIE.dmaBd(<%11 : memref<32xi32, 2>, 0, 32>, 0)
    AIE.useLock(%10, Release, 1)
    cf.br ^bb2
  ^end:
    AIE.end
  }

  %dma = AIE.shimDMA(%t70) {
    %lock1 = AIE.lock(%t70, 1)
      AIE.dmaStart(MM2S0, ^bd0, ^end)
    ^bd0:
      AIE.useLock(%lock1, Acquire, 1)
      AIE.dmaBdPacket(0x6, 10)
      AIE.dmaBd(<%buffer : memref<32xi32>, 0, 32>, 0)
      AIE.useLock(%lock1, Release, 0)
      cf.br ^bd0
    ^end:
      AIE.end
  }

  AIE.packet_flow(0xA) {
    AIE.packet_source<%t70, DMA : 0>
    AIE.packet_dest<%t71,   DMA : 0>
  }
}
