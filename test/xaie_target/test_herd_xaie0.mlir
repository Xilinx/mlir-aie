//===- test_herd_xaie0.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: for (x = ifm_X + 0; x < ifm_X + 1; x += 1) {
// CHECK: for (y = ifm_Y + 0; y < ifm_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 3; x += 1) {
// CHECK: for (y = compute_Y + 0; y < compute_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_SOUTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 3; x < compute_X + 4; x += 1) {
// CHECK: for (y = compute_Y + 0; y < compute_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 1; y < pp_Y + 2; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 3; x < pp_X + 4; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 0; y < compute_Y + 1; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 1; y < compute_Y + 2; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 1; y < compute_Y + 2; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 1; y < compute_Y + 2; y += 1) {
// CHECK: XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE);
// CHECK: }
// CHECK: }

// Phil's sixteen_tiles_plane_daa
// This is a more compact version of what test_herd_routing2.mlir produces
// In other words, this is more of a physical netlist representation, while test_herd_routing2.mlir
// is a logical representation of how to place and route the herds
// Here, we assume that we know the location of herds pp, ifm, and compute:
//   Row Y+2: compute[0][0-3]
//   Row Y+1: ifm[0][0-3]
//   Row Y:   pp[0][0-3]
module @test_herd_xaie0 {
 AIE.device(xcvc1902) {
  %0 = AIE.herd[4][1] { sym_name = "pp" }      // herd ping-pong
  %1 = AIE.herd[4][1] { sym_name = "ifm" }     // herd input-feature-map
  %2 = AIE.herd[4][4] { sym_name = "compute" } // herd compute

  // Route <%1, DMA : 0> to <%2, DMA : 0>
  %ix0 = AIE.iter(0, 1, 1)
  %iy0 = AIE.iter(0, 1, 1)
  %sel0 = AIE.select(%1, %ix0, %iy0)
  AIE.switchbox(%sel0) {
    AIE.connect<DMA : 0, North: 0>
  }

  %ix1 = AIE.iter(0, 3, 1)
  %iy1 = AIE.iter(0, 1, 1)
  %sel1 = AIE.select(%2, %ix1, %iy1)
  AIE.switchbox(%sel1) {
    AIE.connect<North : 0, DMA: 0>
    AIE.connect<North : 0, South: 0>
  }

  %ix2 = AIE.iter(3, 4, 1)
  %iy2 = AIE.iter(0, 1, 1)
  %sel2 = AIE.select(%2, %ix2, %iy2)
  AIE.switchbox(%sel2) {
    AIE.connect<North : 0, DMA: 0>
  }

  // Route <%0, DMA: 0> to <%1, DMA: 1>
  %ix3 = AIE.iter(0, 1, 1)
  %iy3 = AIE.iter(0, 1, 1)
  %sel3 = AIE.select(%0, %ix3, %iy3)
  AIE.switchbox(%sel3) {
    AIE.connect<DMA : 0, North: 0>
  }

  // Route <%0, DMA: 0> to <%1, DMA: 1>
  %ix4 = AIE.iter(0, 1, 1)
  %iy4 = AIE.iter(0, 1, 1)
  %sel4 = AIE.select(%0, %ix4, %iy4)
  AIE.switchbox(%sel4) {
    AIE.connect<DMA  : 0, North : 1>
    AIE.connect<East : 0, North : 2>
    AIE.connect<East : 1, North : 3>
    AIE.connect<East : 2, North : 4>
  }

  %ix5 = AIE.iter(0, 1, 1)
  %iy5 = AIE.iter(1, 2, 1)
  %sel5 = AIE.select(%0, %ix5, %iy5)
  AIE.switchbox(%sel5) {
    AIE.connect<South : 1, North: 1>
    AIE.connect<South : 2, North: 2>
    AIE.connect<South : 3, North: 3>
    AIE.connect<South : 4, North: 4>
  }

  %ix6 = AIE.iter(1, 2, 1)
  %iy6 = AIE.iter(0, 1, 1)
  %sel6 = AIE.select(%0, %ix6, %iy6)
  AIE.switchbox(%sel3) {
    AIE.connect<DMA : 0, West: 0>
    AIE.connect<East : 1, West: 1>
    AIE.connect<East : 2, West: 2>
  }

  %ix7 = AIE.iter(2, 3, 1)
  %iy7 = AIE.iter(0, 1, 1)
  %sel7 = AIE.select(%0, %ix7, %iy7)
  AIE.switchbox(%sel3) {
    AIE.connect<DMA : 0, West: 1>
    AIE.connect<East : 2, West: 2>
  }

  %ix8 = AIE.iter(3, 4, 1)
  %iy8 = AIE.iter(0, 1, 1)
  %sel8 = AIE.select(%0, %ix8, %iy8)
  AIE.switchbox(%sel8) {
    AIE.connect<DMA : 0, West: 2>
  }

  %ix9 = AIE.iter(0, 1, 1)
  %iy9 = AIE.iter(0, 1, 1)
  %sel9 = AIE.select(%2, %ix9, %iy9)
  AIE.switchbox(%sel9) {
    AIE.connect<South : 1, DMA: 1>
    AIE.connect<South : 2, North: 1>
    AIE.connect<South : 3, North: 2>
    AIE.connect<South : 4, North: 3>
  }

  %ix10 = AIE.iter(0, 1, 1)
  %iy10 = AIE.iter(1, 2, 1)
  %sel10 = AIE.select(%2, %ix10, %iy10)
  AIE.switchbox(%sel10) {
    AIE.connect<South : 1, DMA: 1>
    AIE.connect<South : 2, North: 1>
    AIE.connect<South : 3, North: 2>
  }

  %ix11 = AIE.iter(0, 1, 1)
  %iy11 = AIE.iter(2, 3, 1)
  %sel11 = AIE.select(%2, %ix11, %iy11)
  AIE.switchbox(%sel10) {
    AIE.connect<South : 1, DMA: 1>
    AIE.connect<South : 2, North: 1>
  }

  %ix12 = AIE.iter(0, 1, 1)
  %iy12 = AIE.iter(2, 3, 1)
  %sel12 = AIE.select(%2, %ix12, %iy12)
  AIE.switchbox(%sel10) {
    AIE.connect<South : 1, DMA: 1>
  }
 }
}
