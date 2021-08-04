//===- simple_flows2.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = AIE.tile(2, 3)
// CHECK: %[[T22:.*]] = AIE.tile(2, 2)
// CHECK: %[[T11:.*]] = AIE.tile(1, 1)
// CHECK: AIE.flow(%[[T23]], Core : 0, %[[T22]], Core : 1)
// CHECK: AIE.flow(%[[T22]], Core : 0, %[[T11]], Core : 0)

module {
  %t23 = AIE.tile(2, 3)
  %t22 = AIE.tile(2, 2)
  %t11 = AIE.tile(1, 1)
  AIE.flow(%t23, Core : 0, %t22, Core : 1)
  AIE.flow(%t22, Core : 0, %t11, Core : 0)
}