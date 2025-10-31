//===- test_congestion_flow_pktflow_cascade_all_in_one.mlir ----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1) @attention_seg
// CHECK-DAG:   %[[tile_0_0:.*]] = aie.tile(0, 0)
// CHECK-DAG:   %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK-DAG:   %[[tile_2_0:.*]] = aie.tile(2, 0)
// CHECK-DAG:   %[[tile_3_0:.*]] = aie.tile(3, 0)
// CHECK-DAG:   %[[tile_0_1:.*]] = aie.tile(0, 1)
// CHECK-DAG:   %[[tile_1_1:.*]] = aie.tile(1, 1)
// CHECK-DAG:   %[[tile_2_1:.*]] = aie.tile(2, 1)
// CHECK-DAG:   %[[tile_3_1:.*]] = aie.tile(3, 1)


// CHECK:      aie.switchbox(%[[tile_0_0]]) {
// CHECK-NEXT:   aie.connect<South : 3, North : 3>
// CHECK-NEXT:   aie.connect<South : 7, North : 5>
// CHECK-NEXT:   aie.connect<North : 3, South : 2>
// CHECK-NEXT:   %[[v0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(South : 3, %[[v0]]) {keep_pkt_header = true}
// CHECK-NEXT:   aie.packet_rules(North : 2) {
// CHECK-NEXT:     aie.rule(31, 8, %[[v0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK:      aie.switchbox(%[[tile_0_1]]) {
// CHECK-NEXT:   aie.connect<South : 3, DMA : 0>
// CHECK-NEXT:   aie.connect<South : 5, DMA : 1>
// CHECK-NEXT:   aie.connect<DMA : 2, North : 5>
// CHECK-NEXT:   aie.connect<North : 1, DMA : 2>
// CHECK-NEXT:   aie.connect<DMA : 3, South : 3>
// CHECK-NEXT:   %[[v0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   %[[v1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:   aie.masterset(South : 2, %[[v1]])
// CHECK-NEXT:   aie.masterset(North : 1, %[[v0]])
// CHECK-NEXT:   aie.packet_rules(North : 2) {
// CHECK-NEXT:     aie.rule(31, 8, %[[v1]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(27, 0, %[[v0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%[[tile_1_0]]) {
// CHECK-NEXT:   aie.connect<South : 3, North : 1>
// CHECK-NEXT:   aie.connect<South : 7, North : 5>
// CHECK-NEXT:   aie.connect<North : 2, South : 2>
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%[[tile_1_1]]) {
// CHECK-NEXT:   aie.connect<South : 1, DMA : 0>
// CHECK-NEXT:   aie.connect<South : 5, DMA : 1>
// CHECK-NEXT:   aie.connect<DMA : 2, North : 5>
// CHECK-NEXT:   aie.connect<North : 1, DMA : 2>
// CHECK-NEXT:   aie.connect<DMA : 3, South : 2>
// CHECK-NEXT:   %[[v0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(North : 1, %[[v0]])
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(27, 1, %[[v0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%[[tile_2_0]]) {
// CHECK-NEXT:   aie.connect<South : 3, North : 1>
// CHECK-NEXT:   aie.connect<South : 7, North : 5>
// CHECK-NEXT:   aie.connect<North : 2, South : 2>
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%[[tile_2_1]]) {
// CHECK-NEXT:   aie.connect<South : 1, DMA : 0>
// CHECK-NEXT:   aie.connect<South : 5, DMA : 1>
// CHECK-NEXT:   aie.connect<DMA : 2, North : 5>
// CHECK-NEXT:   aie.connect<North : 3, DMA : 2>
// CHECK-NEXT:   aie.connect<DMA : 3, South : 2>
// CHECK-NEXT:   %[[v0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(North : 1, %[[v0]])
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(27, 2, %[[v0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%[[tile_3_0]]) {
// CHECK-NEXT:   aie.connect<South : 3, North : 0>
// CHECK-NEXT:   aie.connect<South : 7, North : 1>
// CHECK-NEXT:   aie.connect<North : 2, South : 2>
// CHECK-NEXT: }
// CHECK:      aie.switchbox(%[[tile_3_1]]) {
// CHECK-NEXT:   aie.connect<South : 0, DMA : 0>
// CHECK-NEXT:   aie.connect<South : 1, DMA : 1>
// CHECK-NEXT:   aie.connect<DMA : 2, North : 5>
// CHECK-NEXT:   aie.connect<North : 1, DMA : 2>
// CHECK-NEXT:   aie.connect<DMA : 3, South : 2>
// CHECK-NEXT:   %[[v0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(North : 1, %[[v0]])
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(27, 3, %[[v0]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
  aie.device(npu1) @attention_seg {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%shim_noc_tile_1_0, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%shim_noc_tile_3_0, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.flow(%shim_noc_tile_0_0, DMA : 1, %mem_tile_0_1, DMA : 1)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_1_1, DMA : 1)
    aie.flow(%shim_noc_tile_2_0, DMA : 1, %mem_tile_2_1, DMA : 1)
    aie.flow(%shim_noc_tile_3_0, DMA : 1, %mem_tile_3_1, DMA : 1)
    aie.packet_flow(0) {
      aie.packet_source<%mem_tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_3, DMA : 0>
      aie.packet_dest<%tile_0_4, DMA : 0>
      aie.packet_dest<%tile_0_5, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%mem_tile_1_1, DMA : 0>
      aie.packet_dest<%tile_1_2, DMA : 0>
      aie.packet_dest<%tile_1_3, DMA : 0>
      aie.packet_dest<%tile_1_4, DMA : 0>
      aie.packet_dest<%tile_1_5, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%mem_tile_2_1, DMA : 0>
      aie.packet_dest<%tile_2_2, DMA : 0>
      aie.packet_dest<%tile_2_3, DMA : 0>
      aie.packet_dest<%tile_2_4, DMA : 0>
      aie.packet_dest<%tile_2_5, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%mem_tile_3_1, DMA : 0>
      aie.packet_dest<%tile_3_2, DMA : 0>
      aie.packet_dest<%tile_3_3, DMA : 0>
      aie.packet_dest<%tile_3_4, DMA : 0>
      aie.packet_dest<%tile_3_5, DMA : 0>
    }
    aie.packet_flow(4) {
      aie.packet_source<%mem_tile_0_1, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_1_2, DMA : 0>
      aie.packet_dest<%tile_2_2, DMA : 0>
      aie.packet_dest<%tile_3_2, DMA : 0>
    }


    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_3_2, DMA : 1)
    aie.packet_flow(5) {
      aie.packet_source<%mem_tile_1_1, DMA : 0>
      aie.packet_dest<%tile_0_3, DMA : 0>
      aie.packet_dest<%tile_1_3, DMA : 0>
      aie.packet_dest<%tile_2_3, DMA : 0>
      aie.packet_dest<%tile_3_3, DMA : 0>
    }
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_1_1, DMA : 2, %tile_3_3, DMA : 1)
    aie.packet_flow(6) {
      aie.packet_source<%mem_tile_2_1, DMA : 0>
      aie.packet_dest<%tile_0_4, DMA : 0>
      aie.packet_dest<%tile_1_4, DMA : 0>
      aie.packet_dest<%tile_2_4, DMA : 0>
      aie.packet_dest<%tile_3_4, DMA : 0>
    }
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_0_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_1_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_2_4, DMA : 1)
    aie.flow(%mem_tile_2_1, DMA : 2, %tile_3_4, DMA : 1)
    aie.packet_flow(7) {
      aie.packet_source<%mem_tile_3_1, DMA : 0>
      aie.packet_dest<%tile_0_5, DMA : 0>
      aie.packet_dest<%tile_1_5, DMA : 0>
      aie.packet_dest<%tile_2_5, DMA : 0>
      aie.packet_dest<%tile_3_5, DMA : 0>
    }
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_0_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_1_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_2_5, DMA : 1)
    aie.flow(%mem_tile_3_1, DMA : 2, %tile_3_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 2)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 2)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 3, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 3, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 3, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 3, %shim_noc_tile_3_0, DMA : 0)
    aie.cascade_flow(%tile_3_5, %tile_3_4)
    aie.cascade_flow(%tile_2_5, %tile_2_4)
    aie.cascade_flow(%tile_1_5, %tile_1_4)
    aie.cascade_flow(%tile_0_5, %tile_0_4)
    aie.cascade_flow(%tile_3_4, %tile_3_3)
    aie.cascade_flow(%tile_2_4, %tile_2_3)
    aie.cascade_flow(%tile_1_4, %tile_1_3)
    aie.cascade_flow(%tile_0_4, %tile_0_3)
    aie.cascade_flow(%tile_3_3, %tile_3_2)
    aie.cascade_flow(%tile_2_3, %tile_2_2)
    aie.cascade_flow(%tile_1_3, %tile_1_2)
    aie.cascade_flow(%tile_0_3, %tile_0_2)

    aie.packet_flow(8) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
  }
}
