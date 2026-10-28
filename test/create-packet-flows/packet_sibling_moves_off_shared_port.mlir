//===- packet_sibling_moves_off_shared_port.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="allow-deadlock-prone=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=STRICT

// Id 24 of (0, 3) DMA:0's tree takes South:3 with (0, 4) Core:0's ids 11 and
// 20, tying id 5 to id 20's arbiter. Taking another South port undoes that;
// splitting ids 5 and 24 apart at (0, 3) would need a fourth South channel,
// and the connects leave three. What remains is a hold cycle through shared
// receivers, as in the model's routing. Reduced from router_mutation.py npu1
// seed 207370212 (fixed-witness).

// WARN-NOT: error
// WARN: warning: Packet flows into receivers they share can deadlock
// STRICT: error: Unable to find a legal routing: Packet flows into receivers they share can deadlock

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %sb_0_1 = aie.switchbox(%t_0_1) {
      aie.connect<North : 0, DMA : 5>
      aie.connect<North : 0, DMA : 1>
    }
    %sb_0_2 = aie.switchbox(%t_0_2) {
      aie.connect<DMA : 0, North : 4>
      aie.connect<North : 1, South : 0>
    }
    %sb_0_3 = aie.switchbox(%t_0_3) {
      aie.connect<South : 4, North : 0>
      aie.connect<Core : 0, South : 1>
    }
    %sb_0_4 = aie.switchbox(%t_0_4) {
    }
    %sb_0_5 = aie.switchbox(%t_0_5) {
      aie.connect<South : 3, DMA : 1>
    }
    aie.packet_flow(24) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%t_0_3, DMA : 0> aie.packet_source<%t_0_2, Core : 0> aie.packet_dest<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_5, Core : 0> }
    aie.packet_flow(14) { aie.packet_source<%t_0_1, DMA : 2> aie.packet_dest<%t_0_4, DMA : 0> }
    aie.packet_flow(28) { aie.packet_source<%t_0_1, DMA : 2> aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_0_0, DMA : 0> }
    aie.packet_flow(11) { aie.packet_source<%t_0_4, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(20) { aie.packet_source<%t_0_4, Core : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(8) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_1, DMA : 0> aie.packet_dest<%t_0_3, Core : 0> }
  }
}
