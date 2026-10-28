//===- badswitchbox-vc1902.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error: 'aie.connect' op illegal shim mux connection

module {
  aie.device(xcvc1902) {
    %30 = aie.tile(3, 0) // Shim-NOC
    aie.shim_mux(%30) {
      aie.connect<DMA: 0, NOC: 0>
    }
  }
}
