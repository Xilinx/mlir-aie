//===- packet_drop_header.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 7;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 0, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x2));
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), NORTH, 0, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x1));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), NORTH, 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), NORTH, 0, {{.*}} 0, {{.*}} XAie_PacketInit(10,0), {{.*}} 0x1F, {{.*}} 1, {{.*}} 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 4));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 4, {{.*}} 0, {{.*}} XAie_PacketInit(3,0), {{.*}} 0x1F, {{.*}} 0, {{.*}} 0));
// CHECK: x = 7;
// CHECK: y = 1;
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), DMA, 0, {{.*}} XAIE_SS_PKT_DROP_HEADER, {{.*}} 0, {{.*}} 0x1));
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 0, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x2));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), DMA, 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), DMA, 0, {{.*}} 0, {{.*}} XAie_PacketInit(10,0), {{.*}} 0x1F, {{.*}} 1, {{.*}} 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), SOUTH, 0, {{.*}} 0, {{.*}} XAie_PacketInit(3,0), {{.*}} 0x1F, {{.*}} 0, {{.*}} 0));

//
// This tests the switchbox configuration lowering for packet switched routing
// to drop headers when the packet's destination is a DMA.
//
module @aie_module {
  aie.device(xcvc1902) {
    %0 = aie.tile(7, 0)
    %1 = aie.switchbox(%0) {
      %7 = aie.amsel<0> (0)
      %8 = aie.amsel<0> (1)
      %9 = aie.masterset(South : 0, %8)
      %10 = aie.masterset(North : 0, %7)
      aie.packet_rules(North : 0) {
        aie.rule(31, 10, %8)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 3, %7)
      }
    }
    %2 = aie.tile(7, 1)
    %3 = aie.switchbox(%2) {
      %7 = aie.amsel<0> (0)
      %8 = aie.amsel<0> (1)
      %9 = aie.masterset(DMA : 0, %7)
      %10 = aie.masterset(South : 0, %8)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 10, %8)
      }
      aie.packet_rules(South : 0) {
        aie.rule(31, 3, %7)
      }
    }
    %4 = aie.lock(%2, 1)
    %5 = aie.buffer(%2) {address = 3072 : i32, sym_name = "buf1"} : memref<16xi32, 2>
    %6 = aie.mem(%2) {
      %7 = aie.dma_start(S2MM, 0, ^bb2, ^bb1)
    ^bb1:  // pred: ^bb0
      %8 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb2:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%4, Acquire, 0)
      aie.dma_bd_packet(2, 3)
      aie.dma_bd(%5 : memref<16xi32, 2>, 0, 16)
      aie.use_lock(%4, Release, 1)
      aie.next_bd ^bb2
    ^bb3:  // 2 preds: ^bb1, ^bb3
      aie.use_lock(%4, Acquire, 1)
      aie.dma_bd_packet(6, 10)
      aie.dma_bd(%5 : memref<16xi32, 2>, 0, 16)
      aie.use_lock(%4, Release, 0)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb1
      aie.end
    }
  }
}
