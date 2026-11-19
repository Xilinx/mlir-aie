//===- packet_port_source_takes_its_wire.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// A packet flow sourced at tile (0,3)'s North : 1 takes the wire tile (0,4)
// drives into it, South : 1, though no hop of its tree crosses it: three
// circuits through rows 4 -> 3 must leave it free.
// CHECK-LABEL: aie.switchbox(%tile_0_4)
// CHECK-NOT:   South : 1>
// CHECK:       }
module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(3) {
      aie.packet_source<%t03, North : 1>
      aie.packet_dest<%t02, DMA : 1>
    }
    aie.flow(%t05, DMA : 0, %t01, DMA : 0)
    aie.flow(%t05, DMA : 1, %t01, DMA : 1)
    aie.flow(%t04, DMA : 0, %t02, DMA : 0)
  }
}
