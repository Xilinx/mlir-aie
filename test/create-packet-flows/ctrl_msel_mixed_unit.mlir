//===- ctrl_msel_mixed_unit.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Priority flow 2 and flow 1 both leave tile (0, 3) by DMA : 0, so they take
// one arbiter. The priority flow takes its highest msel, though flow 1 comes
// first.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: aie.switchbox(%{{.*}}tile_0_3) {
// CHECK-DAG:     %[[LOW:.*]] = aie.amsel<[[ARB:[0-9]]]> (2)
// CHECK-DAG:     %[[HIGH:.*]] = aie.amsel<[[ARB]]> (3)
// CHECK-DAG:     aie.masterset(DMA : 0, %[[LOW]], %[[HIGH]])
// CHECK-DAG:     aie.masterset(North : {{[0-9]}}, %[[LOW]])
// CHECK-DAG:     aie.rule(31, 2, %[[HIGH]])
// CHECK-DAG:     aie.rule(31, 1, %[[LOW]])

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
