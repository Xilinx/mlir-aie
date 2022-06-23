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
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<North : 0, South : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<North : 0, South : 0>
// CHECK:  }
module {
  %t23 = AIE.tile(2, 1)
  %t22 = AIE.tile(2, 0)
  AIE.flow(%t23, North : 0, %t22, PLIO : 0)
}

// -----

// CHECK: module
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK: %[[T21:.*]] = AIE.tile(2, 1)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<North : 0, South : 3>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T20]])  {
// CHECK:    AIE.connect<North : 3, DMA : 1>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T21]])  {
// CHECK:    AIE.connect<Core : 0, South : 0>
// CHECK:  }
module {
  %t20 = AIE.tile(2, 0)
  %t21 = AIE.tile(2, 1)
  AIE.flow(%t21, Core : 0, %t20, DMA : 1)
}

// -----

// CHECK: module
// CHECK: %[[T20:.*]] = AIE.tile(2, 0)
// CHECK: %[[T30:.*]] = AIE.tile(3, 0)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T20]])  {
// CHECK:    AIE.connect<South : 3, East : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T20]])  {
// CHECK:    AIE.connect<DMA : 0, North : 3>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T30]])  {
// CHECK:    AIE.connect<West : 0, South : 3>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T30]])  {
// CHECK:    AIE.connect<North : 3, DMA : 1>
// CHECK:  }
module {
  %t20 = AIE.tile(2, 0)
  %t30 = AIE.tile(3, 0)
  AIE.flow(%t20, DMA : 0, %t30, DMA : 1)
}
