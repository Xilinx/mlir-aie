//===- switchbox-merge.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Two ingress ports sharing an arbiter slot is a merge, which no flow can
// express -- a flow has one source and many destinations, not the other way
// round -- so aie-find-flows leaves it in the switchbox. Read from there, a
// flow arriving at such a port carries on instead of appearing to stop.
//
// Contracting anything is what makes the view read the semantic graph; the
// physical view still draws from the routes, and so does not show this yet.

// REQUIRES: aie-visualize
// RUN: aie-visualize --emit-dot --contract=buffer %s | FileCheck %s

// The flow arrives on East 3 ...
// CHECK-DAG: p_3_2_4_3 -> p_2_2_6_3
// ... and the switchbox carries it on to West 0, which no flow states.
// CHECK-DAG: p_2_2_6_3 -> p_2_2_4_0
// The other ingress sharing the slot reaches the same egress: a merge, which
// is why no single flow can describe this.
// CHECK-DAG: p_2_3_3_1 -> p_2_2_5_1
// CHECK-DAG: p_2_2_5_1 -> p_2_2_4_0

module {
  aie.device(npu2) {
    %t22 = aie.tile(2, 2)
    %t32 = aie.tile(3, 2)
    %t23 = aie.tile(2, 3)

    // A lifted flow that ends on a switchbox port rather than a DMA.
    aie.packet_flow(0) {
      aie.packet_source<%t32, DMA : 0>
      aie.packet_dest<%t22, East : 3>
    } via (%t32 : DMA : 0 -> West : 3)

    // A second flow arriving on the other ingress that shares the slot.
    aie.packet_flow(1) {
      aie.packet_source<%t23, DMA : 0>
      aie.packet_dest<%t22, North : 1>
    } via (%t23 : DMA : 0 -> South : 1)

    // The configuration aie-find-flows could not lift: East 3 and North 1
    // share one arbiter slot, which drives West 0.
    aie.switchbox(%t22) {
      %slot = aie.amsel<1> (0)
      aie.masterset(West : 0, %slot)
      aie.packet_rules(East : 3) {
        aie.rule(31, 0, %slot)
      }
      aie.packet_rules(North : 1) {
        aie.rule(31, 0, %slot)
      }
    }
  }
}
