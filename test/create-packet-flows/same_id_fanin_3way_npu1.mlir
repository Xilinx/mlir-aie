//===- same_id_fanin_3way_npu1.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows=emit-vias=true | aie-opt --aie-split-flow-vias | aie-opt --aie-create-pathfinder-flows | FileCheck %s

// Companion to test/create-flows/same_id_shared_link.mlir, on npu1 and in the
// co-terminal (fan-in) form rather than the divergent form.
//
// Three packet flows share packet id 0 AND share the destination
// (shim_noc_tile_0_0 DMA:0). Same-id flows to the same destination may merge,
// so each tile down column 0 folds its own DMA:0 into the stream from above
// and passes one stream South.
//
// The property under test is that no amsel drives more than one master port.
// One amsel feeding two mastersets is a broadcast: the stream is duplicated,
// every copy is delivered, and the destination over-receives. The pre-#3472
// router did exactly that at tile_0_2 (amsel<1>(0) -> South:1 and South:3).
// Pinning that configuration and running it deadlocks the device, so it is not
// carried as a test; this structural check is the CI-safe form of it.
//
// Fanning IN at a master port is legal and expected: tile_0_2 and tile_0_3
// each let two slave ports drive South through a single amsel. It is the
// reverse direction -- one amsel, several mastersets -- that must never appear.

// CHECK-LABEL: aie.device(npu1)

// CHECK-LABEL: aie.switchbox(%tile_0_2) {
// CHECK-NEXT:   %[[a:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(South : {{[0-9]+}}, %[[a]])
// CHECK-NEXT:   aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[a]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: aie.switchbox(%tile_0_3) {
// CHECK-NEXT:   %[[b:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(South : {{[0-9]+}}, %[[b]])
// CHECK-NEXT:   aie.packet_rules(North : {{[0-9]+}}) {
// CHECK-NEXT:     aie.rule(31, 0, %[[b]])
// CHECK-NEXT:   }
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[b]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK-LABEL: aie.switchbox(%tile_0_4) {
// CHECK-NEXT:   %[[c:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:   aie.masterset(South : {{[0-9]+}}, %[[c]])
// CHECK-NEXT:   aie.packet_rules(DMA : 0) {
// CHECK-NEXT:     aie.rule(31, 0, %[[c]])
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %t0 = aie.tile(0, 2)
    %t1 = aie.tile(0, 3)
    %t2 = aie.tile(0, 4)
    aie.packet_flow(0) {
      aie.packet_source<%t0, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%t1, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
    aie.packet_flow(0) {
      aie.packet_source<%t2, DMA : 0>
      aie.packet_dest<%shim, DMA : 0>
    }
  }
}
