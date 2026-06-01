//===- multi_source_pathfinder.mlir ----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Regression test: when a packet_flow has multiple aie.packet_source ops,
// the pathfinder must route ALL sources to the destination, not just the last
// one encountered (the "last source wins" bug).
//
// Topology: two sources (tiles 1,1 and 2,1) fan-in to one dest (tile 3,1).
//
//   (1,1) --East--> (2,1) --East--> (3,1)
//    src0             src1            dest

// Both source switchboxes must emit packet rules for flow id 0.
// CHECK-LABEL: aie.device(xcvc1902)

// Switchbox at (1,1): must have packet_rules routing Core:0 eastward.
// CHECK:         %[[SW11:.*]] = aie.switchbox(%[[T11:.*]]) {
// CHECK:           aie.packet_rules(Core : 0) {
// CHECK:             aie.rule(31, 0,

// Switchbox at (2,1): must have packet_rules routing Core:0 eastward
// AND forwarding incoming West traffic toward (3,1).
// CHECK:         %[[SW21:.*]] = aie.switchbox(%[[T21:.*]]) {
// CHECK:           aie.packet_rules(Core : 0) {
// CHECK:             aie.rule(31, 0,

// Switchbox at (3,1): must receive and deliver to Core:0.
// CHECK:         %[[SW31:.*]] = aie.switchbox(%[[T31:.*]]) {
// CHECK:           aie.masterset(Core : 0,
// CHECK:           aie.packet_rules(West :

module @multi_source_pathfinder {
  aie.device(xcvc1902) {
    %t11 = aie.tile(1, 1)
    %t21 = aie.tile(2, 1)
    %t31 = aie.tile(3, 1)
    aie.packet_flow(0x0) {
      aie.packet_source<%t11, Core : 0>
      aie.packet_source<%t21, Core : 0>
      aie.packet_dest<%t31, Core : 0>
    }
  }
}
