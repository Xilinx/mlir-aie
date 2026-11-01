//===- ctrl_msel_mixed_unit.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Priority flow 2 and flow 1 both reach (0, 3) DMA : 0. A control-packet
// reload keeps the priority flow's master sets, so flow 1 leaves by DMA : 0
// only alone, by the priority flow's amsel, and reaches it by a slave port of
// its own.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: aie.switchbox(%{{.*}}tile_0_3) {
// CHECK-DAG:     %[[HIGH:.*]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[HIGH]]) {aie.is_ctrl_pkt_overlay}
// CHECK-DAG:     aie.rule(31, 2, %[[HIGH]])
// CHECK-DAG:     aie.rule(31, 1, %[[HIGH]])
// CHECK:       }

module {
  aie.device(npu2_1col) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    aie.packet_flow(1) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%t02, Core : 0>
      aie.packet_dest<%t03, DMA : 0>
    } {priority_route = true}
  }
}
