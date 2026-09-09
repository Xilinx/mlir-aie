//===- arbiter_consumer_placed_first.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// The direct coupling -- one flow ending at the tile the other is emitted from
// -- is symmetric, but which way round the allocator meets it is not. A flow a
// tile emits has its slave port on DMA, and DMA sorts ahead of the North the
// inbound flow arrives on, so the emitted flow is normally placed first and the
// inbound one runs into it. arbiter_deadlock_exhausted.mlir is that order.
//
// Here it is reversed. Flow 9 is a control flow, so it is placed before
// everything, and it ends at the memtile's DMA. Flows 0..4 then fill msel 0 on
// arbiters 0..4, leaving flow 5 -- emitted by that same DMA -- to find arbiter
// 5 already carrying the flow that feeds it. Granting flow 9 while the
// memtile's input buffer is full leaves flow 5, the only thing that drains it,
// waiting on the grant.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %s4 = aie.tile(4, 0)
    %s5 = aie.tile(5, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)

    // Placed first, and consumed by the memtile.
    aie.packet_flow(9) {
      aie.packet_source<%c5, DMA : 0>
      aie.packet_dest<%m, DMA : 0>
    } {priority_route = true}

    // Emitted here, filling msel 0 on arbiters 0..4.
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%s2, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%s3, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s4, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%s5, DMA : 0> }

    // Placed last, and the one that has to share.
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%s0, DMA : 0> }
  }
}

// WARN: warning: at tile (0, 1), packet flow 5 shares arbiter 5 with packet flow 9, which it can deadlock against

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[SHARED:.*]] = aie.amsel<5> (0)
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule(31, 5, %[[SHARED]])
