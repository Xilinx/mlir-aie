//===- test_ps3_xaie.mlir --------------------------------------*- MLIR -*-===//
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
// CHECK: x = 0;
// CHECK: y = 1;
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), DMA, 0, EAST, 0));
// CHECK: x = 1;
// CHECK: y = 1;
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), CORE, 0, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x1));
// CHECK: __mlir_aie_try(XAie_StrmPktSwMstrPortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), CORE, 1, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x3));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), WEST, 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), WEST, 0, {{.*}} 0, {{.*}} XAie_PacketInit(0,0), {{.*}} 0x1F, {{.*}} 0, {{.*}} 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlavePortEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), WEST, 0));
// CHECK: __mlir_aie_try(XAie_StrmPktSwSlaveSlotEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), WEST, 0, {{.*}} 1, {{.*}} XAie_PacketInit(1,0), {{.*}} 0x1F, {{.*}} 1, {{.*}} 0));

// partial multicast
module @test_ps3_xaie {
 aie.device(xcvc1902) {
  %t01 = aie.tile(0, 1)
  %t11 = aie.tile(1, 1)

  aie.switchbox(%t01) {
    aie.connect<DMA : 0, East : 0>
  }

  aie.switchbox(%t11) {
    %a0_0 = aie.amsel<0>(0)
    %a0_1 = aie.amsel<0>(1)

    aie.masterset(Core : 0, %a0_0)
    aie.masterset(Core : 1, %a0_0, %a0_1)

    aie.packet_rules(West : 0) {
      aie.rule(0x1F, 0x0, %a0_0)
      aie.rule(0x1F, 0x1, %a0_1)
    }
  }
 }
}

//module @test_ps3_logical {
//  %t01 = aie.tile(0, 1)
//  %t11 = aie.tile(1, 1)
//
//  aie.packet_flow(0x0) {
//    aie.packet_source<%t01, DMA : 0>
//    aie.packet_dest<%t11, Core : 0>
//    aie.packet_dest<%t11, Core : 1>
//  }
//
//  aie.packet_flow(0x1) {
//    aie.packet_source<%t01, DMA : 0>
//    aie.packet_dest<%t11, Core : 1>
//  }
//}
