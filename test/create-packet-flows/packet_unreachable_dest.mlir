//===- packet_unreachable_dest.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s

// A core tile's DMA:0 cannot drive its own DMA:1 through the switchbox, so the
// stream has to leave the tile and come back, and existing master sets hold
// every port it could leave by. The error names the flow with no path and is
// reported where the flow is declared.

module {
  aie.device(npu1_1col) {
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a2_0 = aie.amsel<2> (0)
      %a3_0 = aie.amsel<3> (0)
      %a4_0 = aie.amsel<4> (0)
      %a5_0 = aie.amsel<5> (0)
      aie.masterset(South : 0, %a2_0)
      aie.masterset(South : 1, %a3_0)
      aie.masterset(South : 2, %a4_0)
      aie.masterset(South : 3, %a5_0)
    }
    // CHECK: packet_unreachable_dest.mlir:[[# @LINE + 1]]:5: error: Unable to find a legal routing: no path leads from (0, 5) DMA:0 to (0, 5) DMA:1 through the connections the switchboxes allow and existing routing leaves free.
    aie.packet_flow(1) { aie.packet_source<%t05, DMA : 0> aie.packet_dest<%t05, DMA : 1> }
  }
}

// -----

// A flow lowered from an ObjectFifo carries the fifo's name, so the error
// names the fifo.

// CHECK: error: loc("of_loop"): Unable to find a legal routing: no path leads from (0, 5) DMA:0 to (0, 5) DMA:1

module {
  aie.device(npu1_1col) {
    %t05 = aie.tile(0, 5)
    %sb05 = aie.switchbox(%t05) {
      %a2_0 = aie.amsel<2> (0)
      %a3_0 = aie.amsel<3> (0)
      %a4_0 = aie.amsel<4> (0)
      %a5_0 = aie.amsel<5> (0)
      aie.masterset(South : 0, %a2_0)
      aie.masterset(South : 1, %a3_0)
      aie.masterset(South : 2, %a4_0)
      aie.masterset(South : 3, %a5_0)
    }
    aie.flow(%t05, DMA : 0, %t05, DMA : 1) loc("of_loop")
  }
}
