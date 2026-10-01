//===- ctrl_pkt_overlay_congested_column.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A design loaded by control packets must route the column control overlay
// the same way the standalone overlay does, or loading it reprograms the
// switches its own control packets travel through (#3837). Here the design's
// circuit flows crowd the memtile's north ports, so one detours through
// column 1, whose tiles the design declares for the overlay to cover them. The
// overlay keeps the master ports, msels and rules it has on its own.

// RUN: aie-opt --pass-pipeline="builtin.module(aie-generate-column-control-overlay{route-shim-to-tile-ctrl=true emit-standalone-overlay=true},aie.device(aie-create-pathfinder-flows))" %s -o %t
// RUN: sed -n '/@design {/,/^  }/p' %t | FileCheck %s
// RUN: sed -n '/@ctrl_pkt_overlay {/,/^  }/p' %t | FileCheck %s

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0) {
// CHECK-DAG:     %[[A50:.+]] = aie.amsel<5> (0)
// CHECK-DAG:     %[[A51:.+]] = aie.amsel<5> (1)
// CHECK-DAG:     %[[A52:.+]] = aie.amsel<5> (2)
// CHECK-DAG:     %[[A53:.+]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(South : 0, %[[A53]]) {is_ctrl_pkt_overlay, keep_pkt_header = true}
// CHECK-DAG:     aie.masterset(North : 1, %[[A52]]) {is_ctrl_pkt_overlay}
// CHECK-DAG:     aie.masterset(North : 3, %[[A51]]) {is_ctrl_pkt_overlay}
// CHECK-DAG:     aie.masterset(TileControl : 0, %[[A50]]) {is_ctrl_pkt_overlay, keep_pkt_header = true}
// CHECK:         aie.packet_rules(South : 7) {
// CHECK-NEXT:      aie.rule(28, 28, %[[A52]]) {priority_route}
// CHECK-NEXT:    } {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(30, 26, %[[A51]]) {priority_route}
// CHECK-NEXT:      aie.rule(31, 15, %[[A50]]) {priority_route}
// CHECK-NEXT:    } {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(TileControl : 0) {
// CHECK-NEXT:      aie.rule(31, 15, %[[A53]]) {priority_route}
// CHECK-NEXT:    } {is_ctrl_pkt_overlay}

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1) {
// CHECK-DAG:     %[[A51:.+]] = aie.amsel<5> (1)
// CHECK-DAG:     %[[A52:.+]] = aie.amsel<5> (2)
// CHECK-DAG:     %[[A53:.+]] = aie.amsel<5> (3)
// CHECK-DAG:     aie.masterset(North : 1, %[[A53]]) {is_ctrl_pkt_overlay}
// CHECK-DAG:     aie.masterset(North : 3, %[[A52]]) {is_ctrl_pkt_overlay}
// CHECK-DAG:     aie.masterset(TileControl : 0, %[[A51]]) {is_ctrl_pkt_overlay, keep_pkt_header = true}
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(28, 28, %[[A53]])
// CHECK-NEXT:    } {is_ctrl_pkt_overlay}
// CHECK:         aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(31, 27, %[[A52]])
// CHECK-NEXT:      aie.rule(31, 26, %[[A51]])
// CHECK-NEXT:    } {is_ctrl_pkt_overlay}

module {
  aie.device(npu2) @design {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    aie.flow(%shim_noc_tile_1_0, DMA : 1, %mem_tile_0_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 1)
    aie.flow(%mem_tile_0_1, DMA : 2, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_3, DMA : 0, %mem_tile_0_1, DMA : 2)
    aie.flow(%mem_tile_0_1, DMA : 3, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_4, DMA : 0, %mem_tile_0_1, DMA : 3)
    aie.flow(%mem_tile_0_1, DMA : 4, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_5, DMA : 0, %mem_tile_0_1, DMA : 4)
    aie.flow(%mem_tile_0_1, DMA : 5, %tile_0_2, DMA : 1)
  }
}
