//===- packet_same_id_trees_fan_in_at_receiver.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Two flows reuse id 24: one from (0, 3), and one from (2, 4)'s DMA and core
// to three receivers, which share only (2, 1) DMA:0 with the first. Joined at
// (2, 2), the trees wait on each other there: a packet of (0, 3)'s can hold
// the arbiter (2, 4)'s need into (2, 2) DMA:0 while it queues behind one of
// theirs into (2, 1). Kept from joining, they come down column 2 on channels
// of their own and fan in at (2, 1) DMA:0 alone.
// Reduced from router_mutation.py's id-reuse mutation of router_properties.py
// npu1 seed 207460302.

// CHECK-LABEL: aie.switchbox(%mem_tile_2_1)
// CHECK:         %[[IN:.*]] = aie.amsel<0> (0)
// CHECK:         aie.masterset(DMA : 0, %[[IN]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 24, %[[IN]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 24, %[[IN]])
// CHECK-NEXT:    }
// CHECK-LABEL: aie.switchbox(%tile_2_2)
// CHECK:         %[[OTHER:.*]] = aie.amsel<0> (0)
// CHECK:         %[[OWN:.*]] = aie.amsel<1> (0)
// CHECK:         aie.masterset(DMA : 0, %[[OWN]])
// CHECK:         aie.masterset(South : 0, %[[OWN]])
// CHECK:         aie.masterset(South : 1, %[[OTHER]])
// CHECK:         aie.packet_rules(North : 1) {
// CHECK-NEXT:      aie.rule(31, 24, %[[OWN]])
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 24, %[[OTHER]])
// CHECK-NEXT:    }

module {
  aie.device(npu1_3col) {
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_3 = aie.tile(2, 3)
    %t_2_4 = aie.tile(2, 4)
    aie.packet_flow(24) { aie.packet_source<%t_0_3, Core : 0> aie.packet_dest<%t_2_1, DMA : 0> aie.packet_dest<%t_2_3, Core : 0> }
    aie.packet_flow(24) { aie.packet_source<%t_2_4, DMA : 0> aie.packet_source<%t_2_4, Core : 0> aie.packet_dest<%t_0_4, DMA : 0> aie.packet_dest<%t_2_1, DMA : 0> aie.packet_dest<%t_2_2, DMA : 0> }
  }
}
