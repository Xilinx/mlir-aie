//===- shadowed_rule.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A switch matches packet rules in order and routes on the first hit, so a rule
// whose ids an earlier rule already claims carries nothing. rule(31, 10) here is
// fully covered by rule(30, 10), which matches {10, 11}: it is dead on hardware
// and must not be reported as a second flow. Related: #437, where the shadowed
// rule points at a *different* master and the misroute is silent.

// RUN: aie-opt -aie-find-flows %s | FileCheck %s
// CHECK: %[[T23:.*]] = aie.tile(2, 3)
// CHECK: %[[T22:.*]] = aie.tile(2, 2)
// CHECK: aie.packet_flow(10) {
// CHECK:   aie.packet_source<%[[T22]], DMA : 0>
// CHECK:   aie.packet_dest<%[[T23]], DMA : 1>
// CHECK: }
// CHECK-NOT: aie.packet_flow
module {
  aie.device(xcvc1902) {
    %tile0 = aie.tile(2, 3)
    %tile1 = aie.tile(2, 2)

    %0 = aie.switchbox(%tile0) {
      %16 = aie.amsel<0> (0)
      %17 = aie.masterset(DMA : 1, %16)
      aie.packet_rules(South : 0) {
        aie.rule(30, 10, %16)
        aie.rule(31, 10, %16)
      }
    }
    %1 = aie.switchbox(%tile1) {
      %18 = aie.amsel<0> (0)
      %19 = aie.masterset(North : 0, %18)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 10, %18)
      }
    }
    aie.wire(%0: Core, %tile0: Core)
    aie.wire(%1: Core, %tile1: Core)
    aie.wire(%0: DMA, %tile0: DMA)
    aie.wire(%1: DMA, %tile1: DMA)
    aie.wire(%0: South, %1: North)
  }
}
