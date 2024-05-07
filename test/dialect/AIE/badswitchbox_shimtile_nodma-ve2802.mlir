//===- badswitchbox-ve2802.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//


// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.connect' op source bundle DMA not supported

module {
  aie.device(xcve2802) {
    %20 = aie.tile(2, 0) // shim-noc tile
    aie.switchbox(%20) {
      aie.connect<DMA: 0, South: 0> // No dma in shimtile.. Go through shim_mux
    }
  }
}
