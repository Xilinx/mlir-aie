//===- find_flows_packet_mask.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A packet rule states a (mask, id) pair, and a lifted flow carries that pair.
//
// The rules a routed design holds are the only record of which ids reach a
// port, because a core may build a packet header at run time. Recovering the
// id alone would narrow the first flow from 0x8..0xb down to 0x8, so routing
// the recovered design would drop the other three ids.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows | FileCheck %s --check-prefix=LIFTED
// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows | aie-opt --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUTED

// The first flow states the rule it came from. The second matches one id, and
// an id states that on its own, so it carries no mask.
// LIFTED-DAG: aie.packet_flow(8 mask 28)
// LIFTED-DAG: aie.packet_flow(0)

// Routing the recovered design rebuilds the same two rules.
// ROUTED: aie.switchbox(%{{.*}}tile_0_2)
// ROUTED:   aie.packet_rules(DMA : 0)
// ROUTED-DAG:     aie.rule(28, 8, %{{.*}})
// ROUTED-DAG:     aie.rule(31, 0, %{{.*}})

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    aie.packet_flow(8 mask 28) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    }

    aie.packet_flow(0) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }
  }
}
