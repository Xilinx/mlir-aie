// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAieTile_CoreControl(&(TileInst[0][1]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: XAieTile_CoreControl(&(TileInst[1][1]), XAIE_ENABLE, XAIE_DISABLE);
// CHECK: // Core Stream Switch column 0 row 1
// CHECK: x = 0;
// CHECK: y = 1;
// CHECK: XAieTile_StrmConnectCct(&(TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_EAST(&(TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_EAST(&(TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: // Core Stream Switch column 1 row 1
// CHECK: x = 1;
// CHECK: y = 1;
// CHECK: XAieTile_StrmConfigMstr(&(TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_MPORT_ME(&(TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE,
// CHECK: 	XAIE_ENABLE,
// CHECK: 	XAIETILE_STRSW_MPORT_CFGPKT(&(TileInst[x][y]),
// CHECK: 		XAIETILE_STRSW_MPORT_ME(&(TileInst[x][y]), 0),
// CHECK: 		XAIE_DISABLE /*drop_header*/,
// CHECK: 		0x1 /*mask*/,
// CHECK: 		0 /*arbiter*/));
// CHECK: XAieTile_StrmConfigSlvSlot(&(TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 0),
// CHECK: 	0 /*slot*/,
// CHECK: 	XAIE_ENABLE,
// CHECK: 	XAIETILE_STRSW_SLVSLOT_CFG(&(TileInst[x][y]),
// CHECK: 		XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 0),
// CHECK: 		0 /*slot*/,
// CHECK: 		0x0 /*ID value*/,
// CHECK: 		0x1F /*mask*/,
// CHECK: 		XAIE_ENABLE,
// CHECK: 		0 /*msel*/,
// CHECK: 		0 /*arbiter*/));
// CHECK: XAieTile_StrmConfigSlvSlot(&(TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 1),
// CHECK: 	0 /*slot*/,
// CHECK: 	XAIE_ENABLE,
// CHECK: 	XAIETILE_STRSW_SLVSLOT_CFG(&(TileInst[x][y]),
// CHECK: 		XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 1),
// CHECK: 		0 /*slot*/,
// CHECK: 		0x1 /*ID value*/,
// CHECK: 		0x1F /*mask*/,
// CHECK: 		XAIE_ENABLE,
// CHECK: 		0 /*msel*/,
// CHECK: 		0 /*arbiter*/));
// many-to-one, single arbiter
module @test_ps2_xaie {
  %t01 = AIE.tile(0, 1)
  %t11 = AIE.tile(1, 1)

  AIE.switchbox(%t01) {
    AIE.connect<DMA : 0, East : 0>
    AIE.connect<DMA : 1, East : 1>
  }

  AIE.switchbox(%t11) {
    %a0_0 = AIE.amsel<0>(0)

    AIE.masterset(ME : 0, %a0_0)

    AIE.packetrules(West : 0) {
      AIE.rule(0x1F, 0x0, %a0_0)
    }

    AIE.packetrules(West : 1) {
      AIE.rule(0x1F, 0x1, %a0_0)
    }
  }
}

//module @test_ps2_logical {
//  %t01 = AIE.tile(0, 1)
//  %t11 = AIE.tile(1, 1)
//
//  AIE.packet_flow(0x0) {
//    AIE.packet_source<%t01, DMA : 0>
//    AIE.packet_dest<%t11, ME : 0>
//  }
//
//  AIE.packet_flow(0x1) {
//    AIE.packet_source<%t01, DMA : 1>
//    AIE.packet_dest<%t11, ME : 0>
//  }
//}
