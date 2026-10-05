//===- find_flows_packet_merge.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Routing gives one rule to several ids that travel together, and narrow rules
// where they part. Ids 9 and 10 share a route up column 0 and separate at
// tile(0,4), so the switchboxes below it carry the enclosing cube of the pair,
// (mask 0x1c, id 0x8).
//
// A recovered flow matching one id must therefore carry no mask: stating the
// full-width mask would make each flow claim its own rule, and the shared
// switchboxes would need two rules where routing used one.

// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows | FileCheck %s --check-prefix=LIFTED
// RUN: aie-opt --aie-create-pathfinder-flows %s | aie-opt --aie-find-flows | aie-opt --aie-create-pathfinder-flows | FileCheck %s --check-prefix=ROUTED

// LIFTED-DAG: aie.packet_flow(9)
// LIFTED-DAG: aie.packet_flow(10)
// LIFTED-NOT: mask

// The rules come back as routing first wrote them: merged below the split,
// narrow above it.
// ROUTED: aie.switchbox(%{{.*}}tile_0_2)
// ROUTED:   aie.rule(28, 8, %{{.*}})
// ROUTED: aie.switchbox(%{{.*}}tile_0_4)
// ROUTED-DAG:   aie.rule(31, 9, %{{.*}})
// ROUTED-DAG:   aie.rule(31, 10, %{{.*}})
// ROUTED: aie.switchbox(%{{.*}}tile_0_5)
// ROUTED:   aie.rule(31, 10, %{{.*}})
// ROUTED: aie.switchbox(%{{.*}}tile_0_3)
// ROUTED:   aie.rule(28, 8, %{{.*}})

module {
  aie.device(npu1_1col) {
    %t02 = aie.tile(0, 2)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    aie.packet_flow(9) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
    }

    aie.packet_flow(10) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t05, DMA : 0>
    }
  }
}
