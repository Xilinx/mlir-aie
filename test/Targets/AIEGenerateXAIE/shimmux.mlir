//===- shimmux.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// CHECK: mlir_aie_configure_switchboxes
// CHECK: x = 2;
// CHECK: y = 0;
// CHECK: __mlir_aie_try(XAie_EnableShimDmaToAieStrmPort(ctx->XAieDevInst, XAie_TileLoc(x,y), 3));

module {
 aie.device(xcvc1902) {
  %t20 = aie.tile(2, 0)
  %mux = aie.shim_mux(%t20)  {
    aie.connect<DMA : 0, North : 3>
  }
 }
}
