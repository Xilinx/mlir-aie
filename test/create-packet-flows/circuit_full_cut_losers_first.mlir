//===- circuit_full_cut_losers_first.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Circuit trees fill the channels from row 1 to row 2; without routing the
// losers first once the overuse stalls, every iteration ends one short.
// Reduced from router_mutation.py npu2 seed 207400143 (port-cap).

module {
  aie.device(npu2) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %t_2_1 = aie.tile(2, 1)
    %t_2_2 = aie.tile(2, 2)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    %t_3_0 = aie.tile(3, 0)
    %t_3_1 = aie.tile(3, 1)
    %t_3_2 = aie.tile(3, 2)
    %t_3_3 = aie.tile(3, 3)
    %t_3_4 = aie.tile(3, 4)
    %t_3_5 = aie.tile(3, 5)
    %t_4_0 = aie.tile(4, 0)
    %t_4_1 = aie.tile(4, 1)
    %t_4_2 = aie.tile(4, 2)
    %t_4_3 = aie.tile(4, 3)
    %t_5_0 = aie.tile(5, 0)
    %t_5_1 = aie.tile(5, 1)
    %t_5_2 = aie.tile(5, 2)
    %t_5_3 = aie.tile(5, 3)
    %t_5_4 = aie.tile(5, 4)
    %t_5_5 = aie.tile(5, 5)
    %t_6_3 = aie.tile(6, 3)
    %t_6_4 = aie.tile(6, 4)
    %t_6_5 = aie.tile(6, 5)
    %t_7_2 = aie.tile(7, 2)
    %t_7_3 = aie.tile(7, 3)
    %t_7_5 = aie.tile(7, 5)
    aie.flow(%t_4_1, DMA : 3, %t_5_2, DMA : 0)
    aie.flow(%t_3_1, DMA : 2, %t_5_4, Core : 0)
    aie.flow(%t_3_0, DMA : 1, %t_2_2, Core : 0)
    aie.flow(%t_5_1, DMA : 2, %t_1_4, DMA : 1)
    aie.flow(%t_5_1, DMA : 1, %t_6_4, Core : 0)
    aie.flow(%t_1_1, DMA : 2, %t_0_2, DMA : 1)
    aie.flow(%t_4_0, DMA : 0, %t_2_4, DMA : 1)
    aie.flow(%t_5_1, DMA : 3, %t_6_3, DMA : 1)
    aie.flow(%t_0_1, DMA : 0, %t_7_5, DMA : 1)
    aie.flow(%t_3_1, DMA : 1, %t_3_2, DMA : 0)
    aie.flow(%t_2_1, DMA : 0, %t_4_2, DMA : 1)
    aie.flow(%t_5_1, DMA : 5, %t_3_3, Core : 0)
    aie.flow(%t_5_1, DMA : 4, %t_3_4, DMA : 0)
    aie.flow(%t_2_1, DMA : 2, %t_1_4, Core : 0)
    aie.flow(%t_0_1, DMA : 2, %t_2_5, DMA : 1)
    aie.flow(%t_0_0, DMA : 1, %t_3_4, DMA : 1)
    aie.flow(%t_3_1, DMA : 0, %t_3_2, Core : 0)
    aie.flow(%t_4_1, DMA : 0, %t_3_2, DMA : 1)
    aie.flow(%t_1_1, DMA : 0, %t_3_5, DMA : 0)
    aie.flow(%t_0_1, DMA : 5, %t_5_5, DMA : 0)
    aie.flow(%t_1_0, DMA : 1, %t_0_4, DMA : 0)
    aie.flow(%t_0_1, DMA : 1, %t_2_5, Core : 0)
    aie.flow(%t_0_1, DMA : 4, %t_7_3, DMA : 0)
    aie.flow(%t_0_1, DMA : 3, %t_2_5, DMA : 0)
    aie.flow(%t_3_1, DMA : 5, %t_7_5, DMA : 0)
    aie.flow(%t_3_1, DMA : 3, %t_2_4, DMA : 0)
    aie.flow(%t_5_0, DMA : 1, %t_5_3, DMA : 1)
    aie.flow(%t_2_1, DMA : 4, %t_3_5, Core : 0)
    aie.flow(%t_2_1, DMA : 5, %t_0_5, Core : 0)
    aie.flow(%t_4_1, DMA : 2, %t_4_3, Core : 0)
    aie.flow(%t_3_1, DMA : 4, %t_6_5, DMA : 0)
    aie.flow(%t_4_1, DMA : 4, %t_5_3, DMA : 0)
    aie.flow(%t_2_1, DMA : 3, %t_5_4, DMA : 1)
    aie.flow(%t_1_1, DMA : 5, %t_4_2, DMA : 0)
    aie.flow(%t_4_1, DMA : 1, %t_7_2, DMA : 0)
    aie.flow(%t_5_0, DMA : 0, %t_1_5, DMA : 0)
    aie.packet_flow(25) { aie.packet_source<%t_1_1, DMA : 3> aie.packet_dest<%t_1_2, DMA : 0> }
  }
}
