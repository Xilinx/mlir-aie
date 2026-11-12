// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-import-overlay --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: aie.device(npu2_1col) @application
// CHECK: %[[A:.*]] = aie.amsel<5> (3) {aie.is_ctrl_pkt_overlay}
// CHECK-NOT: aie.masterset(TileControl
// CHECK: aie.masterset(TileControl : 0, %[[A]]) {aie.is_ctrl_pkt_overlay}
// CHECK-NOT: aie.masterset(TileControl
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 0, %[[A]])
// CHECK-LABEL: aie.device(npu2_1col) @transport

module {
  aie.device(npu2_1col) @application {
    %tile = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, TileControl : 0>
    }
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<5>(3) {aie.is_ctrl_pkt_overlay}
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_pkt_overlay}
      aie.packet_rules(South : 0) {
        aie.rule(31, 15, %a) {aie.is_ctrl_pkt_overlay}
      }
    }
  } {aie.is_ctrl_pkt_overlay}
}
