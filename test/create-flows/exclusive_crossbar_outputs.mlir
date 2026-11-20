//===- exclusive_crossbar_outputs.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// A packet flow that ends at a switchbox port drives that crossbar output but
// takes no wire beyond it, so only the crossbar sees the output in use. A
// circuit takes its output alone: no circuit may be routed onto it. Here three
// circuits cross from row 3 to row 2, which has four South channels, one of
// them the output the packet flow ends at.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:       aie.switchbox(%{{.*}}tile_0_3) {
// CHECK:         aie.masterset(South : 0
// CHECK-NOT:     South : 0>
// CHECK:       }

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t03, South : 0>
    }
    aie.flow(%t04, DMA : 0, %t01, DMA : 0)
    aie.flow(%t04, DMA : 1, %t01, DMA : 1)
    aie.flow(%t05, DMA : 0, %t01, DMA : 2)
  }
}

// -----

// Packets may share an output: two packet flows end at (0, 3) South:0, which
// one master set drives.
// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:       aie.switchbox(%{{.*}}tile_0_3) {
// CHECK-COUNT-1: aie.masterset(South : 0
// CHECK-NOT:     South : 0
// CHECK:       }

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t03, South : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t03, DMA : 1>
      aie.packet_dest<%t03, South : 0>
    }
    aie.flow(%t04, DMA : 0, %t01, DMA : 0)
    aie.flow(%t04, DMA : 1, %t01, DMA : 1)
    aie.flow(%t05, DMA : 0, %t01, DMA : 2)
  }
}
