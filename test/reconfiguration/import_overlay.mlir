// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-import-overlay --aie-create-pathfinder-flows %s -o %t
// RUN: FileCheck %s < %t
// RUN: aie-opt --aie-import-overlay %t | diff %t -

// CHECK-LABEL: aie.device(npu2_1col) @application
// CHECK: %[[CONTROL:.*]] = aie.amsel<5> (3)
// CHECK: aie.masterset(TileControl : 0, %[[CONTROL]]) {aie.is_ctrl_overlay}
// CHECK: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 15, %[[CONTROL]]) {aie.is_ctrl_overlay}
// CHECK-NEXT: aie.rule(31, 0, %{{.*}})
// CHECK-NEXT: }
// CHECK: overlay = @transport
// CHECK-LABEL: aie.device(npu2_1col) @transport

module {
  aie.device(npu2_1col) @application {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<5>(3)
      aie.masterset(TileControl : 0, %a)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 15, %a)
      }
    }
    aie.packet_flow(0) {
      aie.packet_source<%tile, DMA : 0>
      aie.packet_dest<%tile, DMA : 1>
    }
  } {overlay = @transport}
  aie.device(npu2_1col) @transport {
    %tile = aie.tile(0, 2)
    aie.switchbox(%tile) {
      %a = aie.amsel<5>(3) {aie.is_ctrl_overlay}
      aie.masterset(TileControl : 0, %a) {aie.is_ctrl_overlay}
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 15, %a)
      } {aie.is_ctrl_overlay}
    }
  } {aie.is_ctrl_overlay}
}
