//===- routing_check_stall_needs_progress.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows -debug-only=aie-pathfinder %s 2>&1 >/dev/null | FileCheck %s

// Once a routing fits the fabric, an attempt gives up after maxRejectedStall
// iterations without the routing check naming fewer faults or teaching
// something new. A stall drops the trees' id splits every 50 iterations and
// the check teaches the same split again, which is no progress, so these
// attempts give up by the stall rather than run to the iteration limit.
// CHECK-NOT: giving up after {{[0-9]+}} iterations (the iteration limit)
// CHECK: giving up after {{[0-9]+}} iterations (the routing check stopped making progress)
// CHECK-NOT: giving up after {{[0-9]+}} iterations (the iteration limit)

// Rejected routings with overused ones between them, once one has fit,
// stall the same count.
// CHECK: giving up after {{[0-9]+}} iterations (the routing check stopped making progress)
// CHECK-NOT: giving up after {{[0-9]+}} iterations (the iteration limit)

module {
  aie.device(npu2_3col) {
    %t_1_0 = aie.tile(1, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_3 = aie.tile(2, 3)
    aie.packet_flow(17) { aie.packet_source<%t_2_3, DMA : 0> aie.packet_source<%t_1_0, DMA : 1> aie.packet_dest<%t_2_1, DMA : 3> }
    aie.packet_flow(29) { aie.packet_source<%t_2_1, DMA : 3> aie.packet_dest<%t_2_1, DMA : 3> aie.packet_dest<%t_2_3, DMA : 0> }
    aie.packet_flow(23) { aie.packet_source<%t_2_1, DMA : 3> aie.packet_dest<%t_2_3, DMA : 0> }
  }
}

// -----

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_5, Core : 0, %t_0_3, DMA : 1)
    aie.flow(%t_0_5, Core : 0, %t_0_1, DMA : 0)
    aie.packet_flow(9) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_source<%t_0_1, DMA : 2> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(29) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(27) { aie.packet_source<%t_0_2, DMA : 0> aie.packet_dest<%t_0_0, DMA : 1> }
    aie.packet_flow(20) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_1, DMA : 4> }
  }
}
