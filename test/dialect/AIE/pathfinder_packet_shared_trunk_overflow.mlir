//===- pathfinder_packet_shared_trunk_overflow.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Negative control for trunk consolidation: co-sourced control ids {1,15,28}
// and data ids {13,16} share one shim slave, but their minimal covers (control
// 3 rules, data 2 rules) sum to 5, over the 4-slot limit even after
// consolidation. The slot-limit guard must still fire (consolidation does not
// weaken it).

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s --check-prefix=OVERFLOW
// OVERFLOW: slave port packet rules exceed the 4-slot limit

aie.device(npu2) {
  %t00 = aie.tile(0, 0)
  %t01 = aie.tile(0, 1)
  %t02 = aie.tile(0, 2)
  %t03 = aie.tile(0, 3)
  %t04 = aie.tile(0, 4)
  %t05 = aie.tile(0, 5)
  aie.packet_flow(1)  { aie.packet_source<%t00, DMA : 1> aie.packet_dest<%t01, TileControl : 0> } {keep_pkt_header = true, priority_route = true}
  aie.packet_flow(15) { aie.packet_source<%t00, DMA : 1> aie.packet_dest<%t02, TileControl : 0> } {keep_pkt_header = true, priority_route = true}
  aie.packet_flow(28) { aie.packet_source<%t00, DMA : 1> aie.packet_dest<%t03, TileControl : 0> } {keep_pkt_header = true, priority_route = true}
  aie.packet_flow(13) { aie.packet_source<%t00, DMA : 1> aie.packet_dest<%t04, DMA : 0> }
  aie.packet_flow(16) { aie.packet_source<%t00, DMA : 1> aie.packet_dest<%t05, DMA : 0> }
}
