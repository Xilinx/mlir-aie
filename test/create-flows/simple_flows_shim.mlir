//===- simple_flows_shim.mlir ----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s
// CHECK: module
// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK-DAG:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK-DAG:    AIE.connect<North : 0, South : 0>
// CHECK-DAG:  }
// CHECK-DAG:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK-DAG:    AIE.connect<North : 0, South : 0>
// CHECK-DAG:  }
module {
  AIE.device(xcvc1902) {
    %t23 = AIE.tile(2, 1)
    %t22 = AIE.tile(2, 0)
    AIE.flow(%t23, North : 0, %t22, PLIO : 0)
  }
}

// -----

// CHECK: module
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK-DAG:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK-DAG:    AIE.connect<North : 0, South : 3>
// CHECK-DAG:  }
// CHECK-DAG:  %{{.*}} = AIE.shimmux(%[[T20]])  {
// CHECK-DAG:    AIE.connect<North : 3, DMA : 1>
// CHECK-DAG:  }
// CHECK-DAG:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK-DAG:    AIE.connect<Core : 0, South : 0>
// CHECK-DAG:  }
module {
  AIE.device(xcvc1902) {
    %t20 = AIE.tile(2, 0)
    %t21 = AIE.tile(2, 1)
    AIE.flow(%t21, Core : 0, %t20, DMA : 1)
  }
}

// -----

// CHECK: module
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK: %[[T30:.*]] = AIE.tile(3, 0)
// CHECK-DAG:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK-DAG:    AIE.connect<South : 3, East : 0>
// CHECK-DAG:  }
// CHECK-DAG:  %{{.*}} = AIE.shimmux(%[[T20]])  {
// CHECK-DAG:    AIE.connect<DMA : 0, North : 3>
// CHECK-DAG:  }
// CHECK-DAG:  %{{.*}} = AIE.switchbox(%[[T30]])  {
// CHECK-DAG:    AIE.connect<West : 0, South : 3>
// CHECK-DAG:  }
// CHECK-DAG:  %{{.*}} = AIE.shimmux(%[[T30]])  {
// CHECK-DAG:    AIE.connect<North : 3, DMA : 1>
// CHECK-DAG:  }
module {
  AIE.device(xcvc1902) {
    %t20 = AIE.tile(2, 0)
    %t30 = AIE.tile(3, 0)
    AIE.flow(%t20, DMA : 0, %t30, DMA : 1)
  }
}
