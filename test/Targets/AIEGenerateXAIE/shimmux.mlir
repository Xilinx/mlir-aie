//===- shimmux.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
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
