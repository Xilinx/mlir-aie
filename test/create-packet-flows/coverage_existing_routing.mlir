//===- coverage_existing_routing.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Routing already present in the input (an amsel, its masterset, and a wire
// written in the opposite orientation) must be kept and not reused.

// CHECK: %[[SB02:.*]] = aie.switchbox(%{{.*}}tile_0_2) {
// CHECK-NEXT: %[[A0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT: aie.masterset(South : 0, %[[A0]])
// CHECK-NEXT: aie.packet_rules(DMA : 0) {
// CHECK-NEXT: aie.rule(31, 3, %[[A0]])
// CHECK-NOT: aie.amsel<0> (0)
// CHECK-NOT: South : 0
// CHECK: aie.connect<North : 0, South : {{[1-9]}}>
// CHECK-NOT: South : 0
// CHECK: %[[A1:.*]] = aie.amsel<{{.*}}>
// CHECK-NEXT: aie.masterset(South : {{[1-9]}}, %[[A1]])
// CHECK-NEXT: aie.packet_rules(DMA : 1) {
// CHECK-NEXT: aie.rule(31, 5, %[[A1]])
// CHECK: aie.wire(%[[SB02]] : Core, %{{.*}}tile_0_2 : Core)
// CHECK-NOT: aie.wire(%{{.*}}tile_0_2 : Core, %[[SB02]] : Core)

module {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %sb02 = aie.switchbox(%t02) {
      %a0 = aie.amsel<0> (0)
      %m = aie.masterset(South : 0, %a0)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 3, %a0)
      }
    }
    aie.packet_flow(5) {
      aie.packet_source<%t02, DMA : 1>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.wire(%sb02 : Core, %t02 : Core)
    aie.flow(%t03, DMA : 0, %t01, DMA : 1)
  }
}
