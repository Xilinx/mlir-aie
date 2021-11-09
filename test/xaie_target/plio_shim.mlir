//===- plio_shim.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_configure_switchboxes
// CHECK: XAieTile_StrmConnectCct(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_SPORT_SOUTH(&(TileInst[x][y]), 4),
// CHECK:   XAIETILE_STRSW_MPORT_NORTH(&(TileInst[x][y]), 0),
// CHECK:   XAIE_ENABLE);
// CHECK: XAieTile_StrmConnectCct(&(TileInst[x][y]),
// CHECK:   XAIETILE_STRSW_SPORT_NORTH(&(TileInst[x][y]), 0),
// CHECK:   XAIETILE_STRSW_MPORT_SOUTH(&(TileInst[x][y]), 2),
// CHECK:   XAIE_ENABLE);

module {
  %t40 = AIE.tile(4, 0)
  %t41 = AIE.tile(4, 1)
  %4 = AIE.switchbox(%t40)  {
    AIE.connect<South : 4, North : 0>
    AIE.connect<North : 0, South : 2>
  }
  %5 = AIE.switchbox(%t41)  {
    AIE.connect<South : 0, North : 0>
    AIE.connect<North : 0, South : 0>
  }
}
