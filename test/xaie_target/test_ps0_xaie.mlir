// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: XAieTile_CoreControl(&(TileInst[1][1]), XAIE_ENABLE, XAIE_DISABLE);

// CHECK: // Core Stream Switch column 1 row 1
// CHECK: x = 1;
// CHECK: y = 1;

// CHECK: XAieTile_StrmConfigMstr(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_MPORT_CORE(&(TileInst[x][y]), 0),
// CHECK:   XAIE_ENABLE,
// CHECK:   XAIE_ENABLE,
// CHECK:   XAIETILE_STRSW_MPORT_CFGPKT(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_MPORT_CORE(&(TileInst[x][y]), 0),
// CHECK:   XAIE_DISABLE /*drop_header*/,
// CHECK:   0x1 /*mask*/,
// CHECK:   0 /*arbiter*/));

// CHECK: XAieTile_StrmConfigMstr(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_MPORT_CORE(&(TileInst[x][y]), 1),
// CHECK:   XAIE_ENABLE,
// CHECK:   XAIE_ENABLE,
// CHECK:   XAIETILE_STRSW_MPORT_CFGPKT(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_MPORT_CORE(&(TileInst[x][y]), 1),
// CHECK:   XAIE_DISABLE /*drop_header*/,
// CHECK:   0x2 /*mask*/,
// CHECK:   0 /*arbiter*/));

// CHECK: XAieTile_StrmConfigSlvSlot(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 0),
// CHECK:   0 /*slot*/,
// CHECK:   XAIE_ENABLE,
// CHECK:   XAIETILE_STRSW_SLVSLOT_CFG(&(TileInst[x][y]),
// CHECK:     XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 0),
// CHECK:     0 /*slot*/,
// CHECK:     0x0 /*ID value*/,
// CHECK:     0x1F /*mask*/,
// CHECK:     XAIE_ENABLE,
// CHECK:     0 /*msel*/,
// CHECK:     0 /*arbiter*/));

// CHECK: XAieTile_StrmConfigSlvSlot(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 0),
// CHECK:   1 /*slot*/,
// CHECK:   XAIE_ENABLE,
// CHECK:   XAIETILE_STRSW_SLVSLOT_CFG(&(TileInst[x][y]),
// CHECK:     XAIETILE_STRSW_SPORT_WEST(&(TileInst[x][y]), 0),
// CHECK:     1 /*slot*/,
// CHECK:     0x1 /*ID value*/,
// CHECK:     0x1F /*mask*/,
// CHECK:     XAIE_ENABLE,
// CHECK:     1 /*msel*/,
// CHECK:     0 /*arbiter*/));

// one-to-many, single arbiter
module @test_ps0_xaie {
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
