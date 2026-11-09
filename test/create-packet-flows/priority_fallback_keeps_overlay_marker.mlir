//===- priority_fallback_keeps_overlay_marker.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// A prioritized response retains its marker when routing needs the fallback.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK:         aie.masterset(South : 0, %{{.*}}) {aie.priority_route, keep_pkt_header = true}

module {
  aie.device(npu1_2col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    aie.flow(%t_0_1, DMA : 0, %t_1_4, DMA : 1)
    aie.flow(%t_0_2, DMA : 1, %t_0_1, DMA : 1)
    aie.flow(%t_0_2, DMA : 0, %t_0_1, DMA : 0)
    aie.flow(%t_0_4, DMA : 0, %t_1_1, DMA : 2)
    aie.flow(%t_1_2, DMA : 0, %t_0_0, DMA : 0)
    aie.flow(%t_1_5, Core : 0, %t_1_0, DMA : 1)
    aie.flow(%t_1_5, DMA : 1, %t_1_1, DMA : 3)
    aie.flow(%t_0_2, Core : 0, %t_0_0, DMA : 1)
    aie.flow(%t_0_5, Core : 0, %t_1_1, DMA : 0)
    aie.packet_flow(0) { aie.packet_source<%t_1_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> aie.packet_dest<%t_1_2, Core : 0> } {priority_route = true}
    aie.packet_flow(15) { aie.packet_source<%t_0_0, TileControl : 0> aie.packet_dest<%t_0_0, South : 0> } {keep_pkt_header = true, priority_route = true}
  }
}
