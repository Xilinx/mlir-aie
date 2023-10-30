//===- badswitchbox-ve2802.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//


// RUN: not %PYTHON aiecc.py %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'AIE.connect' op source bundle DMA not supported

module {
  AIE.device(xcve2802) {
    %20 = AIE.tile(2, 0) // shim-noc tile
    AIE.switchbox(%20) {
      AIE.connect<DMA: 0, South: 0> // No dma in shimtile.. Go through shimmux
    }
  }
}
