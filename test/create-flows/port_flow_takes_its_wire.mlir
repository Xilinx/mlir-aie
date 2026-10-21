//===- port_flow_takes_its_wire.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-split-flow-vias --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-split-flow-vias --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR

// A flow that ends at a switchbox port takes the wire that port drives, so
// no other flow is routed onto it. --aie-split-flow-vias leaves a via as such
// flows. Here the via holds (0, 3) South:1 and three flows take the other
// three channels from row 3 to row 2.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK-COUNT-1: South : 1>
// CHECK-NOT:     South : 1>
// CHECK:       }

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t04, DMA : 1, %t02, DMA : 1) via (%t04 : DMA : 1 -> South : 1, %t03 : North : 1 -> South : 1, %t02 : North : 1 -> DMA : 1)
    aie.flow(%t04, DMA : 0, %t02, DMA : 0)
    aie.flow(%t05, DMA : 0, %t01, DMA : 0)
    aie.flow(%t05, DMA : 1, %t01, DMA : 1)
  }
}

// -----

// One more flow than the boundary has channels for fails, rather than
// sharing the via's.

// ERR-NOT: error: 'aie.connect'
// ERR:     error: Unable to find a legal routing

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.flow(%t04, DMA : 1, %t02, DMA : 1) via (%t04 : DMA : 1 -> South : 1, %t03 : North : 1 -> South : 1, %t02 : North : 1 -> DMA : 1)
    aie.flow(%t04, DMA : 0, %t02, DMA : 0)
    aie.flow(%t05, DMA : 0, %t01, DMA : 0)
    aie.flow(%t05, DMA : 1, %t01, DMA : 1)
    aie.flow(%t03, DMA : 0, %t01, DMA : 2)
  }
}
