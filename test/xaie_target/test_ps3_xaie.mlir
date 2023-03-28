//===- test_ps3_xaie.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck --match-full-lines %s

// CHECK: XAieTile_CoreControl(&(ctx->TileInst[0][1]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieTile_CoreControl(&(ctx->TileInst[1][1]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: // Core Stream Switch column 0 row 1
// CHECK-NEXT: x = 0;
// CHECK-NEXT: y = 1;
// CHECK-NEXT: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT: 	XAIETILE_STRSW_MPORT_EAST(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT: 	XAIE_ENABLE);
// CHECK-NEXT: // Core Stream Switch column 1 row 1
// CHECK-NEXT: x = 1;
// CHECK-NEXT: y = 1;
// CHECK-NEXT: XAieTile_StrmConfigMstr(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 	XAIETILE_STRSW_MPORT_CORE(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT: 	XAIE_ENABLE,
// CHECK-NEXT: 	XAIE_ENABLE,
// CHECK-NEXT: 	XAIETILE_STRSW_MPORT_CFGPKT(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 		XAIETILE_STRSW_MPORT_CORE(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT: 		XAIE_DISABLE /*drop_header*/,
// CHECK-NEXT: 		0x1 /*mask*/,
// CHECK-NEXT: 		0 /*arbiter*/));
// CHECK-NEXT: XAieTile_StrmConfigMstr(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 	XAIETILE_STRSW_MPORT_CORE(&(ctx->TileInst[x][y]), 1),
// CHECK-NEXT: 	XAIE_ENABLE,
// CHECK-NEXT: 	XAIE_ENABLE,
// CHECK-NEXT: 	XAIETILE_STRSW_MPORT_CFGPKT(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 		XAIETILE_STRSW_MPORT_CORE(&(ctx->TileInst[x][y]), 1),
// CHECK-NEXT: 		XAIE_DISABLE /*drop_header*/,
// CHECK-NEXT: 		0x3 /*mask*/,
// CHECK-NEXT: 		0 /*arbiter*/));
// CHECK-NEXT: XAieTile_StrmConfigSlv(&(ctx->TileInst[x][y]),
// CHECK-NEXT:	  XAIETILE_STRSW_SPORT_WEST(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT:	  XAIE_ENABLE, XAIE_ENABLE);
// CHECK-NEXT: XAieTile_StrmConfigSlvSlot(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 	XAIETILE_STRSW_SPORT_WEST(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT: 	0 /*slot*/,
// CHECK-NEXT: 	XAIE_ENABLE,
// CHECK-NEXT: 	XAIETILE_STRSW_SLVSLOT_CFG(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 		(XAIETILE_STRSW_SPORT_WEST(&(ctx->TileInst[x][y]), 0)),
// CHECK-NEXT: 		0 /*slot*/,
// CHECK-NEXT: 		0x0 /*ID value*/,
// CHECK-NEXT: 		0x1F /*mask*/,
// CHECK-NEXT: 		XAIE_ENABLE,
// CHECK-NEXT: 		0 /*msel*/,
// CHECK-NEXT: 		0 /*arbiter*/));
// CHECK-NEXT: XAieTile_StrmConfigSlv(&(ctx->TileInst[x][y]),
// CHECK-NEXT:	  XAIETILE_STRSW_SPORT_WEST(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT:	  XAIE_ENABLE, XAIE_ENABLE);
// CHECK-NEXT: XAieTile_StrmConfigSlvSlot(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 	XAIETILE_STRSW_SPORT_WEST(&(ctx->TileInst[x][y]), 0),
// CHECK-NEXT: 	1 /*slot*/,
// CHECK-NEXT: 	XAIE_ENABLE,
// CHECK-NEXT: 	XAIETILE_STRSW_SLVSLOT_CFG(&(ctx->TileInst[x][y]),
// CHECK-NEXT: 		(XAIETILE_STRSW_SPORT_WEST(&(ctx->TileInst[x][y]), 0)),
// CHECK-NEXT: 		1 /*slot*/,
// CHECK-NEXT: 		0x1 /*ID value*/,
// CHECK-NEXT: 		0x1F /*mask*/,
// CHECK-NEXT: 		XAIE_ENABLE,
// CHECK-NEXT: 		1 /*msel*/,
// CHECK-NEXT: 		0 /*arbiter*/));

// partial multicast
module @test_ps3_xaie {
 AIE.device(xcvc1902) {
  %t01 = AIE.tile(0, 1)
  %t11 = AIE.tile(1, 1)

  AIE.switchbox(%t01) {
    AIE.connect<DMA : 0, East : 0>
  }

  AIE.switchbox(%t11) {
    %a0_0 = AIE.amsel<0>(0)
    %a0_1 = AIE.amsel<0>(1)

    AIE.masterset(Core : 0, %a0_0)
    AIE.masterset(Core : 1, %a0_0, %a0_1)

    AIE.packetrules(West : 0) {
      AIE.rule(0x1F, 0x0, %a0_0)
      AIE.rule(0x1F, 0x1, %a0_1)
    }
  }
 }
}

//module @test_ps3_logical {
//  %t01 = AIE.tile(0, 1)
//  %t11 = AIE.tile(1, 1)
//
//  AIE.packet_flow(0x0) {
//    AIE.packet_source<%t01, DMA : 0>
//    AIE.packet_dest<%t11, Core : 0>
//    AIE.packet_dest<%t11, Core : 1>
//  }
//
//  AIE.packet_flow(0x1) {
//    AIE.packet_source<%t01, DMA : 0>
//    AIE.packet_dest<%t11, Core : 1>
//  }
//}
