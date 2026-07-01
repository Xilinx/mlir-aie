//===- badswitchbox-ve2802.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'aie.connect' op source bundle FIFO not supported

module {
  aie.device(xcve2802) {
    %01 = aie.tile(0, 1) // mem tile
    aie.switchbox(%01) {
      aie.connect<FIFO: 0, South: 0> // No fifo in memtile
    }
  }
}
