//===- aie2_configure_cascade.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: int mlir_aie_configure_cascade(aie_libxaie_ctx_t* ctx) {
// CHECK: XAie_CoreConfigAccumulatorControl(ctx->XAieDevInst, XAie_TileLoc(1, 3), NORTH, EAST);
// CHECK: XAie_CoreConfigAccumulatorControl(ctx->XAieDevInst, XAie_TileLoc(2, 3), WEST, SOUTH);
// CHECK: XAie_CoreConfigAccumulatorControl(ctx->XAieDevInst, XAie_TileLoc(3, 4), NORTH, SOUTH);
// CHECK: XAie_CoreConfigAccumulatorControl(ctx->XAieDevInst, XAie_TileLoc(3, 3), NORTH, SOUTH);
// CHECK: return XAIE_OK;
// CHECK: } // mlir_aie_configure_cascade

module @cascade_flow {
  aie.device(xcve2802) {
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_3 = aie.tile(3, 3)
    
    aie.configure_cascade(%tile_1_3, North, East)
    aie.configure_cascade(%tile_2_3, West, South)
    aie.configure_cascade(%tile_3_4, North, South)
    aie.configure_cascade(%tile_3_3, North, South)
  }
}
