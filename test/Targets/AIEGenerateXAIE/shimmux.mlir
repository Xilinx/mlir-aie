//===- shim_mux.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 2;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_EnableShimDmaToAieStrmPort(&(ctx->DevInst), XAie_TileLoc(x,y), 3));

module {
 AIE.device(xcvc1902) {
  %t20 = AIE.tile(2, 0)
  %mux = AIE.shim_mux(%t20)  {
    AIE.connect<DMA : 0, North : 3>
  }
 }
}
