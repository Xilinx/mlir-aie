//===- simple2.mlir --------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = aie.tile(2, 3)
// CHECK: %[[T32:.*]] = aie.tile(3, 2)
// CHECK: aie.flow(%[[T23]], Core : 1, %[[T32]], DMA : 0)

module {
  aie.device(xcvc1902) {
    %0 = aie.tile(2, 3)
    %1 = aie.tile(3, 2)
    aie.flow(%0, Core : 1, %1, DMA : 0)
  }
}
