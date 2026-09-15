//===- packet_flow_mask.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A packet flow states the rule it wants with `mask`, and routing keeps that
// rule instead of deriving one from the ids.
//
// A core may build a packet header at run time, so a design can send ids that
// appear nowhere in the IR. Here one flow claims 0x8 through 0xb by masking off
// the low two bits, and routing must leave that claim to it: the rules the
// other two flows get may not match any id in that range.

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// The stated rule reaches the slave port unchanged.
// CHECK: aie.switchbox(%{{.*}}tile_0_2)
// CHECK:   aie.packet_rules(DMA : 0)
// CHECK-DAG:     aie.rule(28, 8, %{{.*}})
// The derived rules match one id each, so none of them reaches into 0x8..0xb.
// CHECK-DAG:     aie.rule(31, 0, %{{.*}})
// CHECK-DAG:     aie.rule(31, 15, %{{.*}})

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)

    // Claims every id that agrees with 0x8 on the bits 0x1c selects.
    aie.packet_flow(8 mask 28) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    }

    aie.packet_flow(0) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }

    aie.packet_flow(15) {
      aie.packet_source<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
    }
  }
}
