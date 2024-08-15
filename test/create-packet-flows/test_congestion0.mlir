//===- test_congestion0.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// REQUIRES: ryzen_ai, chess

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s -o %t.opt
// RUN: FileCheck %s --check-prefix=CHECK1 < %t.opt
// RUN: aie-translate --aie-flows-to-json %t.opt | FileCheck %s --check-prefix=CHECK2

// XFAIL: *

// CHECK1:    %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK1:    %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK1:    %[[VAL_2:.*]] = aie.tile(0, 3)
// CHECK1:    %[[VAL_3:.*]] = aie.tile(0, 4)
// CHECK1:    %[[VAL_4:.*]] = aie.tile(0, 5)
// CHECK1:    aie.packet_flow(0) {
// CHECK1:      aie.packet_source<%[[VAL_1:.*]], DMA : 0>
// CHECK1:      aie.packet_dest<%[[VAL_0:.*]], DMA : 0>
// CHECK1:    }
// CHECK1:    aie.packet_flow(4) {
// CHECK1:      aie.packet_source<%[[VAL_1:.*]], DMA : 1>
// CHECK1:      aie.packet_dest<%[[VAL_0:.*]], DMA : 4>
// CHECK1:    }
// CHECK1:    aie.packet_flow(1) {
// CHECK1:      aie.packet_source<%[[VAL_2:.*]], DMA : 0>
// CHECK1:      aie.packet_dest<%[[VAL_0:.*]], DMA : 1>
// CHECK1:    }
// CHECK1:    aie.packet_flow(2) {
// CHECK1:      aie.packet_source<%[[VAL_3:.*]], DMA : 0>
// CHECK1:      aie.packet_dest<%[[VAL_0:.*]], DMA : 2>
// CHECK1:    }
// CHECK1:    aie.packet_flow(3) {
// CHECK1:      aie.packet_source<%[[VAL_4:.*]], DMA : 0>
// CHECK1:      aie.packet_dest<%[[VAL_0:.*]], DMA : 3>
// CHECK1:    }

// CHECK2: "total_path_length": 8

module {
 aie.device(npu1_1col) {
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)

  aie.packet_flow(0) { 
    aie.packet_source<%tile_0_2, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 0>
  }
  aie.packet_flow(1) { 
    aie.packet_source<%tile_0_3, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 1>
  }
  aie.packet_flow(2) { 
    aie.packet_source<%tile_0_4, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 2>
  }
  aie.packet_flow(3) { 
    aie.packet_source<%tile_0_5, DMA : 0> 
    aie.packet_dest<%tile_0_1, DMA : 3>
  }
  aie.packet_flow(4) { 
    aie.packet_source<%tile_0_2, DMA : 1> 
    aie.packet_dest<%tile_0_1, DMA : 4>
  }
 }
}
