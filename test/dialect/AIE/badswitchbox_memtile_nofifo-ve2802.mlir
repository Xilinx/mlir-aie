//===- badswitchbox-ve2802.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//


// RUN: not aiecc.py %s |& FileCheck %s
// CHECK: error{{.*}}'AIE.connect' op source bundle FIFO not supported

module {
  AIE.device(xcve2802) {
    %01 = AIE.tile(0, 1) // mem tile
    AIE.switchbox(%01) {
      AIE.connect<FIFO: 0, South: 0> // No fifo in memtile
    }
  }
}
