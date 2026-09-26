//===- unused_amsel_reserved.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// An amsel that no master set uses, as aie-find-flows can leave behind, still
// has a rule steering id 31 to it, so the router must not take the same
// arbiter and msel for id 5 at DMA : 1.

// CHECK-LABEL: aie.switchbox(%tile_0_3) {
// CHECK-NEXT:    %[[PINNED:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:    aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(31, 31, %[[PINNED]])
// CHECK-NEXT:    }
// CHECK-NOT:     aie.amsel<0> (0)
// CHECK:         %[[NEW:.*]] = aie.amsel
// CHECK-NEXT:    aie.masterset(DMA : 1, %[[NEW]])
// CHECK:         aie.rule(31, 5, %[[NEW]])

module {
  aie.device(npu2_1col) {
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %sb = aie.switchbox(%t_0_3) {
      %a = aie.amsel<0> (0)
      aie.packet_rules(South : 3) {
        aie.rule(31, 31, %a)
      }
    }
    aie.packet_flow(5) {
      aie.packet_source<%t_0_2, DMA : 0>
      aie.packet_dest<%t_0_3, DMA : 1>
    }
  }
}
