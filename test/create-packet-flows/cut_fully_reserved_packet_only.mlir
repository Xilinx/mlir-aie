//===- cut_fully_reserved_packet_only.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s

// CHECK: error: Unable to find a legal routing: the packet flows cross from row 2 to row 3, which takes 1 channel across, and it has 0 channels.

// cut_fully_reserved.mlir's design, with the crossing flow a packet flow
// instead of a circuit one: no circuit flow crosses the boundary, so the
// message must not say "the circuit flows from" with nothing to name.

module {
  aie.device(npu1_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %sb_0_2 = aie.switchbox(%t_0_2) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
      aie.connect<South : 2, North : 2>
      aie.connect<South : 3, North : 3>
      aie.connect<Core : 0, North : 4>
      aie.connect<DMA : 0, North : 5>
    }
    aie.packet_flow(3) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_4, DMA : 0> }
  }
}
