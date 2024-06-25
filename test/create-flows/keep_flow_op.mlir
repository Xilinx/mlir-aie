//===- keep_flow_op.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows='keep-flow-op=true' %s | FileCheck %s
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
// CHECK:  aie.flow(%[[T20:.*]], DMA : 0, %[[T30:.*]], DMA : 1)
module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t30 = aie.tile(3, 0)
    aie.flow(%t20, DMA : 0, %t30, DMA : 1)
  }
}
