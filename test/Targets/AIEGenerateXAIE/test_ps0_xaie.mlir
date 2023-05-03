//===- test_ps0_xaie.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 1;
// CHECK: y = 1;
// CHECK: XAie_StrmPktSwMstrPortEnable(&(ctx->DevInst), XAie_TileLoc(x,y), CORE, 0, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x1);
// CHECK: XAie_StrmPktSwMstrPortEnable(&(ctx->DevInst), XAie_TileLoc(x,y), CORE, 1, {{.*}} XAIE_SS_PKT_DONOT_DROP_HEADER, {{.*}} 0, {{.*}} 0x2);
// CHECK: XAie_StrmPktSwSlavePortEnable(&(ctx->DevInst), XAie_TileLoc(x,y), WEST, 0);
// CHECK: XAie_StrmPktSwSlaveSlotEnable(&(ctx->DevInst), XAie_TileLoc(x,y), WEST, 0, {{.*}} 0, {{.*}} XAie_PacketInit(0,0), {{.*}} 0x1F, {{.*}} 0, {{.*}} 0);
// CHECK: XAie_StrmPktSwSlavePortEnable(&(ctx->DevInst), XAie_TileLoc(x,y), WEST, 0);
// CHECK: XAie_StrmPktSwSlaveSlotEnable(&(ctx->DevInst), XAie_TileLoc(x,y), WEST, 0, {{.*}} 1, {{.*}} XAie_PacketInit(1,0), {{.*}} 0x1F, {{.*}} 1, {{.*}} 0);

// one-to-many, single arbiter
module @test_ps0_xaie {
 AIE.device(xcvc1902) {
  %t11 = AIE.tile(1, 1)

  AIE.switchbox(%t11) {
    %a0_0 = AIE.amsel<0>(0)
    %a0_1 = AIE.amsel<0>(1)

    AIE.masterset(Core : 0, %a0_0)
    AIE.masterset(Core : 1, %a0_1)

    AIE.packetrules(West : 0) {
      AIE.rule(0x1F, 0x0, %a0_0)
      AIE.rule(0x1F, 0x1, %a0_1)
    }
  }
 }
}

//module @test_ps0_logical {
//  %t11 = AIE.tile(1, 1)
//
//  AIE.packet_flow(0x0) {
//    AIE.packet_source<%t11, West : 0>
//    AIE.packet_dest<%t11, Core : 0>
//  }
//
//  AIE.packet_flow(0x1) {
//    AIE.packet_source<%t11, West : 0>
//    AIE.packet_dest<%t11, Core : 1>
//  }
//}
