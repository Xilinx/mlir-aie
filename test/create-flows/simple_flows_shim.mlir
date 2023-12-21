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
// CHECK: %[[T21:.*]] = aie.tile(2, 1)
// CHECK: %[[T20:.*]] = aie.tile(2, 0)
// CHECK:  %{{.*}} = aie.switchbox(%[[T20]])  {
// CHECK:    aie.connect<North : 0, South : 0>
// CHECK:  }
// CHECK:  %{{.*}} = aie.switchbox(%[[T21]])  {
// CHECK:    aie.connect<North : 0, South : 0>
// CHECK:  }
module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 1)
    %t22 = aie.tile(2, 0)
    aie.flow(%t23, North : 0, %t22, PLIO : 0)
  }
}

// -----

// CHECK: module
// CHECK: %[[T20:.*]] = aie.tile(2, 0)
// CHECK: %[[T21:.*]] = aie.tile(2, 1)
// CHECK:  %{{.*}} = aie.switchbox(%[[T20]])  {
// CHECK:    aie.connect<North : 0, South : 3>
// CHECK:  }
// CHECK:  %{{.*}} = aie.shim_mux(%[[T20]])  {
// CHECK:    aie.connect<North : 3, DMA : 1>
// CHECK:  }
// CHECK:  %{{.*}} = aie.switchbox(%[[T21]])  {
// CHECK:    aie.connect<Core : 0, South : 0>
// CHECK:  }
module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t21 = aie.tile(2, 1)
    aie.flow(%t21, Core : 0, %t20, DMA : 1)
  }
}

// -----

// CHECK: module
// CHECK: %[[T20:.*]] = aie.tile(2, 0)
// CHECK: %[[T30:.*]] = aie.tile(3, 0)
// CHECK:  %{{.*}} = aie.switchbox(%[[T20]])  {
// CHECK:    aie.connect<South : 3, East : 0>
// CHECK:  }
// CHECK:  %{{.*}} = aie.shim_mux(%[[T20]])  {
// CHECK:    aie.connect<DMA : 0, North : 3>
// CHECK:  }
// CHECK:  %{{.*}} = aie.switchbox(%[[T30]])  {
// CHECK:    aie.connect<West : 0, South : 3>
// CHECK:  }
// CHECK:  %{{.*}} = aie.shim_mux(%[[T30]])  {
// CHECK:    aie.connect<North : 3, DMA : 1>
// CHECK:  }
module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t30 = aie.tile(3, 0)
    aie.flow(%t20, DMA : 0, %t30, DMA : 1)
  }
}
