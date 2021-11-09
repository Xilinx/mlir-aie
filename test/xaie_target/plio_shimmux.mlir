//===- plio_shimmux.mlir ---------------------------------------*- MLIR -*-===//
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
// CHECK: XAieTile_ShimStrmDemuxConfig(&(TileInst[x][y]),
// CHECK:        XAIETILE_SHIM_STRM_DEM_SOUTH2,
// CHECK:        XAIETILE_SHIM_STRM_DEM_PL);

module {
  %t20 = AIE.tile(2, 0)
  %t21 = AIE.tile(2, 1)
  %4 = AIE.switchbox(%t20)  {
    AIE.connect<North : 0, South : 2>
  }
  %5 = AIE.switchbox(%t21)  {
    AIE.connect<North : 0, South : 0>
  }
  %6 = AIE.shimmux(%t20)  {
    AIE.connect<North : 2, PLIO : 2>
  }
}
