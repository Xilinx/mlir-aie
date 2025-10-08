//===- packet_shim_header.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_shimdma_70
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaSetPkt(&([[bd0]]), XAie_PacketInit(10,6)));

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 7;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), NORTH, 0, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x1));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 3));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 3, {{.*}} 0, {{.*}} XAie_PacketInit(10,0), {{.*}} 0x1F, {{.*}} 0, {{.*}} 0));
// CHECK: x = 7;
// CHECK: y = 1;
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), DMA, 0, {{.*}} XAIE_SS_PKT_DROP_HEADER, {{.*}} 0, {{.*}} 0x1));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 0, {{.*}} 0, {{.*}} XAie_PacketInit(10,0), {{.*}} 0x1F, {{.*}} 0, {{.*}} 0));
// CHECK: x = 7;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_EnableShimDmaToAieStrmPort(ctx->XAieDevInst, XAie_TileLoc(x,y), 3));

//
// This tests the shim DMA BD configuration lowering for packet switched routing
// to insert packet headers for shim DMA BDs.
//
module @aie_module {
  aie.device(xcvc1902) {
    %0 = aie.tile(7, 0)
    %1 = aie.shim_mux(%0) {
      aie.connect<DMA : 0, North : 3>
    }
    %2 = aie.switchbox(%0) {
      %10 = aie.amsel<0> (0)
      %11 = aie.masterset(North : 0, %10)
      aie.packet_rules(South : 3) {
        aie.rule(31, 10, %10)
      }
    }
    %3 = aie.tile(7, 1)
    %4 = aie.switchbox(%3) {
      %10 = aie.amsel<0> (0)
      %11 = aie.masterset(DMA : 0, %10)
      aie.packet_rules(South : 0) {
        aie.rule(31, 10, %10)
      }
    }
    %5 = aie.lock(%3, 1)
    %6 = aie.buffer(%3) {address = 3072 : i32, sym_name = "buf1"} : memref<32xi32, 2>
    %7 = aie.external_buffer {sym_name = "buf"} : memref<32xi32>
    %8 = aie.mem(%3) {
      %10 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%5, Acquire, 0)
      aie.dma_bd(%6 : memref<32xi32, 2>, 0, 32)
      aie.use_lock(%5, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
    %9 = aie.shim_dma(%0) {
      %10 = aie.lock(%0, 1)
      %11 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%10, Acquire, 1)
      aie.dma_bd_packet(6, 10)
      aie.dma_bd(%7 : memref<32xi32>, 0, 32)
      aie.use_lock(%10, Release, 0)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      aie.end
    }
  }
}
