//===- packet_multicast_parallel_channels.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Four streams go down from (0,3) to (0,2), which has four channels, so each
// takes one. Packet id 3 from (0,5) is delivered to two ports of (0,2); taking
// a second channel for the second of them leaves one stream without, and
// discounting the hops its tree already takes kept it doing so. Reduced from
// router_mutation.py seed 180.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK-COUNT-1: aie.rule(31, 3,
// CHECK-NOT:     aie.rule(31, 3,
// CHECK:       aie.switchbox

module {
  aie.device(npu2_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_5, DMA : 1, %t_0_2, DMA : 0)
    aie.flow(%t_0_5, Core : 0, %t_0_1, DMA : 0)
    aie.packet_flow(3) {
      aie.packet_source<%t_0_1, DMA : 5>
      aie.packet_source<%t_0_5, DMA : 0>
      aie.packet_dest<%t_0_2, Core : 0>
      aie.packet_dest<%t_0_2, DMA : 1>
    }
    aie.packet_flow(0) {
      aie.packet_source<%t_0_4, DMA : 1>
      aie.packet_dest<%t_0_1, DMA : 3>
    }
    aie.packet_flow(18) {
      aie.packet_source<%t_0_1, DMA : 4>
      aie.packet_dest<%t_0_1, DMA : 4>
    }
    aie.packet_flow(6) {
      aie.packet_source<%t_0_1, DMA : 4>
      aie.packet_dest<%t_0_2, Core : 0>
    }
  }
}
