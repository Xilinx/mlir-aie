//===- aie2_cascade_flow.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: int mlir_aie_configure_cascade(aie_libxaie_ctx_t* ctx) {
// CHECK: XAie_CoreConfigAccumulatorControl(&(ctx->DevInst), XAie_TileLoc(1, 3), NORTH, EAST);
// CHECK: XAie_CoreConfigAccumulatorControl(&(ctx->DevInst), XAie_TileLoc(2, 3), WEST, SOUTH);
// CHECK: XAie_CoreConfigAccumulatorControl(&(ctx->DevInst), XAie_TileLoc(3, 4), NORTH, SOUTH);
// CHECK: XAie_CoreConfigAccumulatorControl(&(ctx->DevInst), XAie_TileLoc(3, 3), NORTH, SOUTH);
// CHECK: return XAIE_OK;
// CHECK: } // mlir_aie_configure_cascade

module @cascade_flow {
  aie.device(xcve2802) {
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_3 = aie.tile(3, 3)
    %cascade_switchbox_1_3 = aie.cascade_switchbox(%tile_1_3) {
      aie.connect<North : 0, East : 0>
    }
    %cascade_switchbox_2_3 = aie.cascade_switchbox(%tile_2_3) {
      aie.connect<West : 0, South : 0>
    }
    %cascade_switchbox_3_4 = aie.cascade_switchbox(%tile_3_4) {
      aie.connect<North : 0, South : 0>
    }
    %cascade_switchbox_3_3 = aie.cascade_switchbox(%tile_3_3) {
      aie.connect<North : 0, South : 0>
    }
  }
}
