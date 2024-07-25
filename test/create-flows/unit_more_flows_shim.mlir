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

// RUN: aie-opt --aie-create-pathfinder-flows -split-input-file %s | FileCheck %s

// CHECK-LABEL: test70
// CHECK: %[[T70:.*]] = aie.tile(7, 0)
// CHECK: %[[T71:.*]] = aie.tile(7, 1)
// CHECK:  %[[SB70:.*]] = aie.switchbox(%[[T70]])  {
// CHECK:    aie.connect<North : 0, South : 2>
// CHECK:  }
// CHECK:  %[[SH70:.*]] = aie.shim_mux(%[[T70]])  {
// CHECK:    aie.connect<North : 2, PLIO : 2>
// CHECK:  }
// CHECK:  %[[SB71:.*]] = aie.switchbox(%[[T71]])  {
// CHECK:    aie.connect<North : 0, South : 0>
// CHECK:  }

// Tile 7,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams to PLIO 2,3,4,5
module @test70 {
  aie.device(xcvc1902) {
    %t70 = aie.tile(7, 0)
    %t71 = aie.tile(7, 1)
    aie.flow(%t71, North : 0, %t70, PLIO : 2)
  }
}

// -----

// CHECK-LABEL: test60
// CHECK: %[[T60:.*]] = aie.tile(6, 0)
// CHECK: %[[T61:.*]] = aie.tile(6, 1)
// CHECK:  %[[SB60:.*]] = aie.switchbox(%[[T60]])  {
// CHECK:    aie.connect<South : 6, North : 0>
// CHECK:  }
// CHECK:  %[[SH60:.*]] = aie.shim_mux(%[[T60]])  {
// CHECK:    aie.connect<PLIO : 6, North : 6>
// CHECK:  }
// CHECK:  %[[SB61:.*]] = aie.switchbox(%[[T61]])  {
// CHECK:    aie.connect<South : 0, DMA : 1>
// CHECK:  }

// Tile 6,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams from PLIO 2,3,6,7
module @test60 {
  aie.device(xcvc1902) {
    %t60 = aie.tile(6, 0)
    %t61 = aie.tile(6, 1)
    aie.flow(%t60, PLIO : 6, %t61, DMA : 1)
  }
}

// -----

// CHECK-LABEL: test40
// CHECK: %[[T40:.*]] = aie.tile(4, 0)
// CHECK: %[[T41:.*]] = aie.tile(4, 1)
// CHECK:  %[[SB40:.*]] = aie.switchbox(%[[T40]])  {
// CHECK:    aie.connect<North : 0, South : 3>
// CHECK:    aie.connect<South : 4, North : 0>
// CHECK:  }
// CHECK:  %[[SB41:.*]] = aie.switchbox(%[[T41]])  {
// CHECK:    aie.connect<North : 0, South : 0>
// CHECK:    aie.connect<South : 0, North : 0>
// CHECK:  }

// Tile 4,0 is a shim PL tile and does not contain a ShimMux.
module @test40 {
  aie.device(xcvc1902) {
    %t40 = aie.tile(4, 0)
    %t41 = aie.tile(4, 1)
    aie.flow(%t41, North : 0, %t40, PLIO : 3)
    aie.flow(%t40, PLIO : 4, %t41, North : 0)
  }
}

// -----

// CHECK-LABEL: test100
// CHECK: %[[T100:.*]] = aie.tile(10, 0)
// CHECK: %[[T101:.*]] = aie.tile(10, 1)
// CHECK:  %[[SB100:.*]] = aie.switchbox(%[[T100]])  {
// CHECK:    aie.connect<North : 0, South : 4>
// CHECK:  }
// CHECK:  %[[SH100:.*]] = aie.shim_mux(%[[T100]])  {
// CHECK:    aie.connect<North : 4, NOC : 2>
// CHECK:  }
// CHECK:  %[[SB101:.*]] = aie.switchbox(%[[T101]])  {
// CHECK:    aie.connect<North : 0, South : 0>
// CHECK:  }

// Tile 10,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams to NOC 0,1,2,3
module @test100 {
  aie.device(xcvc1902) {
    %t100 = aie.tile(10, 0)
    %t101 = aie.tile(10, 1)
    aie.flow(%t101, North : 0, %t100, NOC : 2)
  }
}

