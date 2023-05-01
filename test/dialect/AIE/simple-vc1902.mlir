//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// CHECK: %[[T01:.*]] = AIE.tile(0, 1)
// CHECK: %[[T12:.*]] = AIE.tile(1, 2)
// CHECK: AIE.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)

module {
  AIE.device(xcvc1902) {
    %01 = AIE.tile(0, 1)
    %12 = AIE.tile(1, 2)
    %02 = AIE.tile(0, 2)
    %lock = AIE.lock(%12, 15) { sym_name = "lock1" }
    AIE.flow(%01, DMA : 0, %12, Core : 1)
  }
}
