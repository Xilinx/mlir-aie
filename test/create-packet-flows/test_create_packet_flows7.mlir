//===- test_create_packet_flows0.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// generating tile op declarations for tiles on the packetflow
// CHECK-LABEL:   module @test_create_packet_flows7 {
// CHECK:        %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:          %[[VAL0:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(South : 0, %[[VAL0]])
// CHECK:          aie.packet_rules(Trace : 0) {
// CHECK:            aie.rule(31, 1, %[[VAL0]])
// CHECK:          }
// CHECK:        }
// CHECK:        %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:        %{{.*}} = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK:          aie.connect<North : 3, DMA : 1>
// CHECK:        }
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:          %[[VAL1:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(South : 3, %[[VAL1]]) {keep_pkt_header = true}
// CHECK:          aie.packet_rules(East : 0) {
// CHECK:            aie.rule(31, 1, %[[VAL1]])
// CHECK:          }
// CHECK:        }
// CHECK:        aie.packet_flow(1) {
// CHECK:          aie.packet_source<%[[TILE_1_2]], Trace : 0>
// CHECK:          aie.packet_dest<%[[TILE_0_0]], DMA : 1>
// CHECK:        } {keep_pkt_header = true}
// CHECK:        %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:          %[[VAL2:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(West : 0, %[[VAL2]])
// CHECK:          aie.packet_rules(North : 0) {
// CHECK:            aie.rule(31, 1, %[[VAL2]])
// CHECK:          }
// CHECK:        }
// CHECK:        %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:        %{{.*}} = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:          %[[VAL3:.*]] = aie.amsel<0> (0)
// CHECK:          %{{.*}} = aie.masterset(South : 0, %[[VAL3]])
// CHECK:          aie.packet_rules(North : 0) {
// CHECK:            aie.rule(31, 1, %[[VAL3]])
// CHECK:          }
// CHECK:        }
module @test_create_packet_flows7 {
  aie.device(npu2) {
    %tile_1_2 = aie.tile(1, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.packet_flow(1) {
      aie.packet_source<%tile_1_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
  }
}

