//===- more_flows_shim.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

//
// These tests verify pathfinder routing flows to/from PLIO in shim tiles.  
//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s
// CHECK: module
// CHECK: %[[T70:.*]] = AIE.tile(7, 0)
// CHECK: %[[T71:.*]] = AIE.tile(7, 1)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T70]])  {
// CHECK:    AIE.connect<North : 0, South : 2>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T70]])  {
// CHECK:    AIE.connect<North : 2, PLIO : 2>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T71]])  {
// CHECK:    AIE.connect<North : 0, South : 0>
// CHECK:  }

// Tile 7,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams to PLIO 2,3,4,5
module {
  AIE.device(xcvc1902) {
    %t70 = AIE.tile(7, 0)
    %t71 = AIE.tile(7, 1)
    AIE.flow(%t71, North : 0, %t70, PLIO : 2)
  }
}

// -----

// CHECK: module
// CHECK: %[[T60:.*]] = AIE.tile(6, 0)
// CHECK: %[[T61:.*]] = AIE.tile(6, 1)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T60]])  {
// CHECK:    AIE.connect<South : 6, North : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T60]])  {
// CHECK:    AIE.connect<PLIO : 6, North : 6>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T61]])  {
// CHECK:    AIE.connect<South : 0, DMA : 1>
// CHECK:  }

// Tile 6,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams from PLIO 2,3,6,7
module {
  AIE.device(xcvc1902) {
    %t60 = AIE.tile(6, 0)
    %t61 = AIE.tile(6, 1)
    AIE.flow(%t60, PLIO : 6, %t61, DMA : 1)
  }
}

// -----

// CHECK: module
// CHECK: %[[T40:.*]] = AIE.tile(4, 0)
// CHECK: %[[T41:.*]] = AIE.tile(4, 1)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T40]])  {
// CHECK:    AIE.connect<North : 0, South : 3>
// CHECK:    AIE.connect<South : 4, North : 0>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T41]])  {
// CHECK:    AIE.connect<North : 0, South : 0>
// CHECK:    AIE.connect<South : 0, North : 0>
// CHECK:  }

// Tile 4,0 is a shim PL tile and does not contain a ShimMux.
module {
  AIE.device(xcvc1902) {
    %t40 = AIE.tile(4, 0)
    %t41 = AIE.tile(4, 1)
    AIE.flow(%t41, North : 0, %t40, PLIO : 3)
    AIE.flow(%t40, PLIO : 4, %t41, North : 0)
  }
}

// -----

// CHECK: module
// CHECK: %[[T100:.*]] = AIE.tile(10, 0)
// CHECK: %[[T101:.*]] = AIE.tile(10, 1)
// CHECK:  %{{.*}} = AIE.switchbox(%[[T100]])  {
// CHECK:    AIE.connect<North : 0, South : 4>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.shimmux(%[[T100]])  {
// CHECK:    AIE.connect<North : 4, NOC : 2>
// CHECK:  }
// CHECK:  %{{.*}} = AIE.switchbox(%[[T101]])  {
// CHECK:    AIE.connect<North : 0, South : 0>
// CHECK:  }

// Tile 10,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams to NOC 0,1,2,3
module {
  AIE.device(xcvc1902) {
    %t100 = AIE.tile(10, 0)
    %t101 = AIE.tile(10, 1)
    AIE.flow(%t101, North : 0, %t100, NOC : 2)
  }
}

