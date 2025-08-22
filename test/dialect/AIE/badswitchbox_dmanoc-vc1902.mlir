//===- badswitchbox-vc1902.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//


// RUN: not %PYTHON aiecc.py %s 2>&1 | FileCheck %s
// CHECK: error:
// XFAIL: *

module {
  aie.device(xcvc1902) {
    %30 = aie.tile(3, 0) // Shim-NOC
    aie.shim_mux(%30) {
      // Can't connect DMA and NOC in same tile.
      aie.connect<DMA: 0, North: 3>
      aie.connect<NOC: 1, North: 6>
    }
  }
}
