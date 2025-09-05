//===- plio_shimmux.mlir ---------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

//
// This tests the lowering from aie.switchbox ops to configuration register
// writes for LibXAIEV1. This test targets NoC shim tiles that must configure
// stream switches, and for some PLIOs the shim_mux, to connect AIE array
// streams to PL.
//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 2;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), NORTH, 0, SOUTH, 2));
// CHECK: x = 2;
// CHECK: y = 1;
// CHECK: __mlir_aie_try(XAie_StrmConnCctEnable(ctx->XAieDevInst, XAie_TileLoc(x,y), NORTH, 0, SOUTH, 0));
// CHECK: x = 2;
// CHECK: y = 0;

module {
 aie.device(xcvc1902) {
  %t20 = aie.tile(2, 0)
  %t21 = aie.tile(2, 1)
  %4 = aie.switchbox(%t20)  {
    aie.connect<North : 0, South : 2>
  }
  %5 = aie.switchbox(%t21)  {
    aie.connect<North : 0, South : 0>
  }
  %6 = aie.shim_mux(%t20)  {
    aie.connect<North : 2, PLIO : 2>
  }
 }
}
