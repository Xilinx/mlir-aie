//===- packet_flow_mask_overlap_reroute.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Flow 19 (mask 19) and flow 24 (mask 28) both claim id 31. On one slave port
// they would need rules that match it for different master ports, so they have
// to enter every switchbox they share on different ports, which they can: the
// router steers them apart instead of failing. Reduced from
// router_mutation.py seed 385.

// CHECK-LABEL: aie.switchbox(%tile_1_5)
// CHECK:           aie.rule(28, 24,
// CHECK-NEXT:    }
// CHECK-NEXT:    aie.packet_rules(South : {{[0-5]}}) {
// CHECK-NEXT:      aie.rule(19, 19,
// CHECK-NEXT:    }

module {
  aie.device(npu2_3col) {
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_5 = aie.tile(1, 5)
    %sb_1_2 = aie.switchbox(%t_1_2) {
      aie.connect<West : 3, North : 5>
    }
    aie.packet_flow(19, mask = 19) {
      aie.packet_source<%t_1_1, DMA : 2>
      aie.packet_dest<%t_1_5, Core : 0>
    }
    aie.packet_flow(24, mask = 28) {
      aie.packet_source<%t_1_3, Core : 0>
      aie.packet_dest<%t_1_5, Core : 0>
    }
    aie.packet_flow(13) {
      aie.packet_source<%t_1_2, Core : 0>
      aie.packet_dest<%t_1_5, Core : 0>
    }
    aie.packet_flow(21) {
      aie.packet_source<%t_1_2, Core : 0>
      aie.packet_dest<%t_1_1, DMA : 0>
    }
    aie.packet_flow(30) {
      aie.packet_source<%t_1_2, Core : 0>
      aie.packet_dest<%t_1_3, DMA : 0>
    }
  }
}
