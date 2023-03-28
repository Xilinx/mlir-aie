//===- shimmux.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_switchboxes
// CHECK: XAieTile_ShimStrmMuxConfig(&(ctx->TileInst[x][y]),
// CHECK:        XAIETILE_SHIM_STRM_MUX_SOUTH3,
// CHECK:        XAIETILE_SHIM_STRM_MUX_DMA);

module {
 AIE.device(xcvc1902) {
  %t20 = AIE.tile(2, 0)
  %mux = AIE.shimmux(%t20)  {
    AIE.connect<DMA : 0, North : 3>
  }
 }
}
