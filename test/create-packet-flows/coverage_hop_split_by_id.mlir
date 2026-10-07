//===- coverage_hop_split_by_id.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Tile (0,3) has one free arbiter for two master ports, so the router tries
// to circuit switch its hops. Flows 1 and 2 share a slave port there but leave
// by different master ports, so that hop must stay packet switched.

// CHECK-LABEL: aie.switchbox(%tile_0_3)
// CHECK-NOT:     aie.connect
// CHECK:         aie.packet_rules(South : {{[0-9]+}}) {
// CHECK-DAG:       aie.rule(31, 1, %{{.*}})
// CHECK-DAG:       aie.rule(31, 2, %{{.*}})
// CHECK:         }
// CHECK-NOT:     aie.connect

module {
  aie.device(npu1_2col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t13 = aie.tile(1, 3)
    %sb03 = aie.switchbox(%t03) {
      %a1 = aie.amsel<1> (0)
      %a2 = aie.amsel<2> (0)
      %a3 = aie.amsel<3> (0)
      %a4 = aie.amsel<4> (0)
      %a5 = aie.amsel<5> (0)
      %m1 = aie.masterset(DMA : 0, %a1)
      %m2 = aie.masterset(DMA : 1, %a2)
      %m3 = aie.masterset(Core : 0, %a3)
      %m4 = aie.masterset(North : 2, %a4)
      %m5 = aie.masterset(North : 3, %a5)
    }
    aie.packet_flow(1) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t13, DMA : 0>
    }
  }
}
