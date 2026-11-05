//===- relax_ladder_skips_replays.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts
// RUN: aie-opt --aie-create-pathfinder-flows -debug-only=aie-pathfinder %s 2>&1 >/dev/null | FileCheck %s --check-prefix=SKIP
// RUN: aie-opt --aie-create-pathfinder-flows -aie-pathfinder-verify-replays -debug-only=aie-pathfinder %s 2>&1 >/dev/null | FileCheck %s --check-prefix=VERIFY
// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Routing the ids apart turns channel sharing off again, and no attempt
// consults it, so the next two attempts replay the first two exactly.
// SKIP: Relax: route ids apart
// SKIP-NOT: Begin findPaths iteration
// SKIP: this attempt replays attempt 0 of the ladder, so it is skipped
// SKIP: Relax: share channels
// SKIP-NOT: Begin findPaths iteration
// SKIP: this attempt replays attempt 1 of the ladder, so it is skipped
// SKIP: Relax: route cluster member 1 first

// VERIFY: this attempt replays attempt 0 of the ladder, as verified
// VERIFY: this attempt replays attempt 1 of the ladder, as verified

// CHECK: %switchbox_0_1 = aie.switchbox(%mem_tile_0_1)
// CHECK: %switchbox_0_5 = aie.switchbox(%tile_0_5)

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    aie.packet_flow(9) { aie.packet_source<%t_0_1, DMA : 4> aie.packet_dest<%t_0_1, DMA : 4> aie.packet_dest<%t_0_5, Core : 0> }
    aie.packet_flow(10) { aie.packet_source<%t_0_1, DMA : 4> aie.packet_dest<%t_0_4, DMA : 0> }
    aie.packet_flow(14, mask = 14) { aie.packet_source<%t_0_4, Core : 0> aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_3, Core : 0> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(26) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_1, DMA : 1> aie.packet_dest<%t_0_3, Core : 0> aie.packet_dest<%t_0_5, DMA : 1> }
  }
}
