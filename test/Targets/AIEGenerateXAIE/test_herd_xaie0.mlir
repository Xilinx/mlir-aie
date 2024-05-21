//===- test_herd_xaie0.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// REQUIRES: stephenn
// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: for (x = ifm_X + 0; x < ifm_X + 1; x += 1) {
// CHECK: for (y = ifm_Y + 0; y < ifm_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 3; x += 1) {
// CHECK: for (y = compute_Y + 0; y < compute_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_SOUTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 3; x < compute_X + 4; x += 1) {
// CHECK: for (y = compute_Y + 0; y < compute_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 1; y < pp_Y + 2; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 0; x < pp_X + 1; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_EAST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = pp_X + 3; x < pp_X + 4; x += 1) {
// CHECK: for (y = pp_Y + 0; y < pp_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_DMA(&(ctx->TileInst[x][y]), 0),
// CHECK: 	XAIETILE_STRSW_MPORT_WEST(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 0; y < compute_Y + 1; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 4),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 1; y < compute_Y + 2; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 3),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 1; y < compute_Y + 2; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 2),
// CHECK: 	XAIETILE_STRSW_MPORT_NORTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
// CHECK: }
// CHECK: }
// CHECK: for (x = compute_X + 0; x < compute_X + 1; x += 1) {
// CHECK: for (y = compute_Y + 1; y < compute_Y + 2; y += 1) {
// CHECK: __mlir_aie_try(XAieTile_StrmConnectCct(&(ctx->TileInst[x][y]),
// CHECK: 	XAIETILE_STRSW_SPORT_SOUTH(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIETILE_STRSW_MPORT_DMA(&(ctx->TileInst[x][y]), 1),
// CHECK: 	XAIE_ENABLE));
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
 aie.device(xcvc1902) {
  %0 = aie.herd[4][1] { sym_name = "pp" }      // herd ping-pong
  %1 = aie.herd[4][1] { sym_name = "ifm" }     // herd input-feature-map
  %2 = aie.herd[4][4] { sym_name = "compute" } // herd compute

  // Route <%1, DMA : 0> to <%2, DMA : 0>
  %ix0 = aie.iter(0, 1, 1)
  %iy0 = aie.iter(0, 1, 1)
  %sel0 = aie.select(%1, %ix0, %iy0)
  aie.switchbox(%sel0) {
    aie.connect<DMA : 0, North: 0>
  }

  %ix1 = aie.iter(0, 3, 1)
  %iy1 = aie.iter(0, 1, 1)
  %sel1 = aie.select(%2, %ix1, %iy1)
  aie.switchbox(%sel1) {
    aie.connect<North : 0, DMA: 0>
    aie.connect<North : 0, South: 0>
  }

  %ix2 = aie.iter(3, 4, 1)
  %iy2 = aie.iter(0, 1, 1)
  %sel2 = aie.select(%2, %ix2, %iy2)
  aie.switchbox(%sel2) {
    aie.connect<North : 0, DMA: 0>
  }

  // Route <%0, DMA: 0> to <%1, DMA: 1>
  %ix3 = aie.iter(0, 1, 1)
  %iy3 = aie.iter(0, 1, 1)
  %sel3 = aie.select(%0, %ix3, %iy3)
  aie.switchbox(%sel3) {
    aie.connect<DMA : 0, North: 0>
  }

  // Route <%0, DMA: 0> to <%1, DMA: 1>
  %ix4 = aie.iter(0, 1, 1)
  %iy4 = aie.iter(0, 1, 1)
  %sel4 = aie.select(%0, %ix4, %iy4)
  aie.switchbox(%sel4) {
    aie.connect<DMA  : 0, North : 1>
    aie.connect<East : 0, North : 2>
    aie.connect<East : 1, North : 3>
    aie.connect<East : 2, North : 4>
  }

  %ix5 = aie.iter(0, 1, 1)
  %iy5 = aie.iter(1, 2, 1)
  %sel5 = aie.select(%0, %ix5, %iy5)
  aie.switchbox(%sel5) {
    aie.connect<South : 1, North: 1>
    aie.connect<South : 2, North: 2>
    aie.connect<South : 3, North: 3>
    aie.connect<South : 4, North: 4>
  }

  %ix6 = aie.iter(1, 2, 1)
  %iy6 = aie.iter(0, 1, 1)
  %sel6 = aie.select(%0, %ix6, %iy6)
  aie.switchbox(%sel3) {
    aie.connect<DMA : 0, West: 0>
    aie.connect<East : 1, West: 1>
    aie.connect<East : 2, West: 2>
  }

  %ix7 = aie.iter(2, 3, 1)
  %iy7 = aie.iter(0, 1, 1)
  %sel7 = aie.select(%0, %ix7, %iy7)
  aie.switchbox(%sel3) {
    aie.connect<DMA : 0, West: 1>
    aie.connect<East : 2, West: 2>
  }

  %ix8 = aie.iter(3, 4, 1)
  %iy8 = aie.iter(0, 1, 1)
  %sel8 = aie.select(%0, %ix8, %iy8)
  aie.switchbox(%sel8) {
    aie.connect<DMA : 0, West: 2>
  }

  %ix9 = aie.iter(0, 1, 1)
  %iy9 = aie.iter(0, 1, 1)
  %sel9 = aie.select(%2, %ix9, %iy9)
  aie.switchbox(%sel9) {
    aie.connect<South : 1, DMA: 1>
    aie.connect<South : 2, North: 1>
    aie.connect<South : 3, North: 2>
    aie.connect<South : 4, North: 3>
  }

  %ix10 = aie.iter(0, 1, 1)
  %iy10 = aie.iter(1, 2, 1)
  %sel10 = aie.select(%2, %ix10, %iy10)
  aie.switchbox(%sel10) {
    aie.connect<South : 1, DMA: 1>
    aie.connect<South : 2, North: 1>
    aie.connect<South : 3, North: 2>
  }

  %ix11 = aie.iter(0, 1, 1)
  %iy11 = aie.iter(2, 3, 1)
  %sel11 = aie.select(%2, %ix11, %iy11)
  aie.switchbox(%sel10) {
    aie.connect<South : 1, DMA: 1>
    aie.connect<South : 2, North: 1>
  }

  %ix12 = aie.iter(0, 1, 1)
  %iy12 = aie.iter(2, 3, 1)
  %sel12 = aie.select(%2, %ix12, %iy12)
  aie.switchbox(%sel10) {
    aie.connect<South : 1, DMA: 1>
  }
 }
}
