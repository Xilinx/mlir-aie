//===- routing_check_pinned_faults_stop.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts
// RUN: aie-opt --aie-create-pathfinder-flows=allow-deadlock-prone=true -debug-only=aie-pathfinder %s 2>&1 >/dev/null | FileCheck %s --check-prefix=STOP
// RUN: aie-opt --aie-create-pathfinder-flows=allow-deadlock-prone=true %s | FileCheck %s

// With the prioritized trees pinned, the routing check rejects the same hold
// cycle through connections only those trees take, which no penalty moves, so
// each attempt stops at once rather than after maxRejectedStall rejections.
// STOP-NOT: the routing check stopped making progress
// STOP: the routing check names only connections pinned trees take (4) after 1 iterations
// STOP-NOT: the routing check stopped making progress
// STOP: the routing check names only connections pinned trees take (4) after 1 iterations
// STOP-NOT: the routing check stopped making progress

// The design still routes.
// CHECK: %switchbox_8_3 = aie.switchbox(%tile_8_3)
// CHECK: aie.packet_rules(Core : 0)
// CHECK: %switchbox_8_5 = aie.switchbox(%tile_8_5)

module {
  aie.device(xcvc1902) {
    %t_7_5 = aie.tile(7, 5)
    %t_8_1 = aie.tile(8, 1)
    %t_8_3 = aie.tile(8, 3)
    %t_8_4 = aie.tile(8, 4)
    %t_8_5 = aie.tile(8, 5)
    %t_8_7 = aie.tile(8, 7)
    %t_8_8 = aie.tile(8, 8)
    %sb_8_4 = aie.switchbox(%t_8_4) {
      aie.connect<East : 0, North : 0>
    }
    aie.packet_flow(1) { aie.packet_source<%t_8_7, Core : 1> aie.packet_dest<%t_7_5, DMA : 0> aie.packet_dest<%t_8_5, Core : 0> }
    aie.packet_flow(1) { aie.packet_source<%t_8_3, Core : 0> aie.packet_dest<%t_8_5, DMA : 0> } {priority_route = true}
    aie.packet_flow(19) { aie.packet_source<%t_8_1, DMA : 1> aie.packet_dest<%t_8_5, DMA : 0> aie.packet_dest<%t_8_8, DMA : 1> } {priority_route = true}
  }
}
