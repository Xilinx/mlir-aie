//===- plio_shim.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

//
// This tests the lowering from AIE.switchbox ops to configuration register
// writes for LibXAIEV1. This test targets PL shim tiles that only contain
// stream switches that connect the AIE array to PL.
//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 4;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), SOUTH, 4, NORTH, 0));
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2));
// CHECK: x = 4;
// CHECK: y = 1;
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), SOUTH, 0, NORTH, 0));
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(&(ctx->DevInst), XAie_TileLoc(x,y), NORTH, 0, SOUTH, 0));

module {
 AIE.device(xcvc1902) {
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
}
