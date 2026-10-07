//===- subcube_cover_overbudget.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// RUN: not aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 | FileCheck %s

// Five destinations on one slave port need five rules, over the 4-slot budget.
// Existing connections leave (0,2) one link down to the memtile and the
// memtile no link south, so no packets can reach the memtile by a second port.

// CHECK: error: Unable to find a legal routing: at tile (0, 1), the packet flows entering on North:0 need 5 packet rules, and a slave port holds 4.

module @overbudget {
  aie.device(npu1_1col) {
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %sb01 = aie.switchbox(%t01) {
      aie.connect<DMA : 0, South : 0>
      aie.connect<DMA : 1, South : 1>
      aie.connect<DMA : 2, South : 2>
      aie.connect<DMA : 3, South : 3>
    }
    %sb02 = aie.switchbox(%t02) {
      aie.connect<DMA : 1, South : 1>
      aie.connect<Core : 0, South : 2>
      aie.connect<North : 0, South : 3>
    }
    aie.packet_flow(0x0) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(0x1) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 1> }
    aie.packet_flow(0x2) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 2> }
    aie.packet_flow(0x3) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 3> }
    aie.packet_flow(0x4) { aie.packet_source<%t02, DMA : 0> aie.packet_dest<%t01, DMA : 4> }
  }
}
